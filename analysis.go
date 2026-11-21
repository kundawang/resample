// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package resample

import (
	"fmt"
	"math"
)

// maxAnalysisFFT bounds the work Analyze performs. The prototype transform
// is zero padded to at least twice the filter length, so a filter longer
// than half this cannot be analyzed and Analyze reports an error instead of
// attempting an unbounded allocation.
const maxAnalysisFFT = 1 << 20

// Bands configures the frequency ranges over which Analyze measures the
// filter response. Frequencies are in hertz.
//
// The zero value is not used directly: when every field is zero, Analyze
// substitutes defaults derived from the Quality and the two rates. An
// explicitly supplied Bands must have two non-empty, non-overlapping ranges,
// with the passband below the stopband, and every edge at or below the
// higher of the two Nyquist frequencies.
type Bands struct {
	// PassbandLow and PassbandHigh bound the region measured for ripple.
	PassbandLow, PassbandHigh float64

	// StopbandLow and StopbandHigh bound the region measured for alias and
	// image rejection.
	StopbandLow, StopbandHigh float64
}

// QualityReport characterizes the anti-imaging and anti-aliasing filter of a
// conversion. Every figure is computed from the exact filter that New would
// build for the same rates and Quality, so repeated calls with the same
// inputs return identical numbers.
type QualityReport struct {
	// PassbandRippleDB is the largest deviation of the passband magnitude
	// from unit gain, in dB, always non-negative.
	PassbandRippleDB float64

	// StopbandAttenuationDB is the smallest stopband attenuation, in dB:
	// the level of the strongest stopband component relative to unit input.
	StopbandAttenuationDB float64

	// Taps is the number of multiply-accumulate operations per output
	// sample, identical to Resampler.Taps.
	Taps int

	// Delay is the filter's group delay in input samples. It is compensated
	// internally, so output sample k corresponds to input time k*m/l with
	// no residual offset; identical to Resampler.Delay.
	Delay int
}

// Analyze reports the measured quality of the filter used for a conversion
// from inRate to outRate at Quality q, without processing any audio.
//
// The response is measured on a fixed frequency grid by a bounded-size FFT
// of the prototype coefficients, so the result is deterministic and the
// cost is capped at maxAnalysisFFT points. The streaming Write and Flush
// outputs are unaffected.
//
// When bands is the zero value, the passband runs from DC to
// (1-q.Transition) times the lower Nyquist and the stopband from
// (1+q.Transition/2) times the lower Nyquist to the higher Nyquist. The
// transition band in between belongs to neither and is excluded from both
// figures.
func Analyze(inRate, outRate float64, q Quality, bands Bands) (*QualityReport, error) {
	if err := q.validate(); err != nil {
		return nil, err
	}
	l, m, err := ratio(inRate, outRate)
	if err != nil {
		return nil, err
	}

	// Estimate the filter length with the same formulas designPolyphase
	// uses, and reject a design that would exceed the analysis budget before
	// allocating it. This keeps extreme ratios failing with a clear error
	// rather than an out-of-memory panic.
	nyq := 0.5 / float64(max(l, m))
	transition := nyq * q.Transition
	nEst := kaiserLength(q.Attenuation, transition)
	dEst := (nEst - 1 + 2*l - 1) / (2 * l)
	if dEst < 1 {
		dEst = 1
	}
	protoLenEst := float64(2*dEst+1) * float64(l)
	if !(protoLenEst > 0) || math.IsInf(protoLenEst, 0) ||
		2*protoLenEst > float64(maxAnalysisFFT) {
		return nil, fmt.Errorf("resample: filter for ratio %d/%d at the requested quality is too long to analyze (budget %d points)",
			l, m, maxAnalysisFFT)
	}

	r, err := New(inRate, outRate, q)
	if err != nil {
		return nil, err
	}

	domainHi := math.Max(inRate, outRate) / 2
	if bands == (Bands{}) {
		nyqLow := math.Min(inRate, outRate) / 2
		stopbandLow := nyqLow * (1 + q.Transition/2)
		if stopbandLow >= domainHi {
			return nil, fmt.Errorf("resample: rates %g and %g leave no room for a default stopband; specify explicit Bands",
				inRate, outRate)
		}
		bands = Bands{
			PassbandHigh: nyqLow * (1 - q.Transition),
			StopbandLow:  stopbandLow,
			StopbandHigh: domainHi,
		}
	}
	if err := bands.validate(domainHi); err != nil {
		return nil, err
	}

	// Reconstruct the prototype filter from the phases New actually built.
	// Phase p holds prototype indices p, p+l, p+2l, ...
	protoLen := r.taps * l
	proto := make([]float64, protoLen)
	for p, ph := range r.phases {
		for j, v := range ph {
			proto[p+j*l] = v
		}
	}

	nfft := 1
	for nfft < 2*protoLen {
		nfft <<= 1
	}
	if nfft > maxAnalysisFFT {
		return nil, fmt.Errorf("resample: analysis needs %d points, exceeding the budget of %d",
			nfft, maxAnalysisFFT)
	}

	// H(e^jw) of the prototype. Bin k is frequency k*l*inRate/nfft Hz,
	// covering 0..l*inRate/2, which contains both Nyquist frequencies.
	re := make([]float64, nfft)
	im := make([]float64, nfft)
	for i, v := range proto {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("resample: non-finite filter coefficient at index %d", i)
		}
		re[i] = v
	}
	fft(re, im)

	binOf := func(hz float64) int {
		return int(math.Round(hz * float64(nfft) / (float64(l) * inRate)))
	}
	pbLo, pbHi := binOf(bands.PassbandLow), binOf(bands.PassbandHigh)
	sbLo, sbHi := binOf(bands.StopbandLow), binOf(bands.StopbandHigh)
	if pbLo > pbHi {
		return nil, fmt.Errorf("resample: passband %g..%g Hz contains no analysis points",
			bands.PassbandLow, bands.PassbandHigh)
	}
	if sbLo > sbHi {
		return nil, fmt.Errorf("resample: stopband %g..%g Hz contains no analysis points",
			bands.StopbandLow, bands.StopbandHigh)
	}

	magnitude := func(k int) (float64, error) {
		// The prototype's DC magnitude is l: each phase sums to one, and
		// |H_proto(Omega)|/l equals the per-phase response magnitude, the
		// gain a tone at that frequency sees through the resampler.
		mag := math.Hypot(re[k], im[k]) / float64(l)
		if math.IsNaN(mag) || math.IsInf(mag, 0) || mag <= 0 {
			return 0, fmt.Errorf("resample: non-finite frequency response at %g Hz",
				float64(k)*float64(l)*inRate/float64(nfft))
		}
		return mag, nil
	}

	// Ripple: worst passband deviation from the unit nominal gain.
	var ripple float64
	for k := pbLo; k <= pbHi; k++ {
		mag, err := magnitude(k)
		if err != nil {
			return nil, err
		}
		ripple = math.Max(ripple, math.Abs(20*math.Log10(mag)))
	}

	// Attenuation: the strongest stopband component sets the worst case.
	worst := 0.0
	for k := sbLo; k <= sbHi; k++ {
		mag, err := magnitude(k)
		if err != nil {
			return nil, err
		}
		worst = math.Max(worst, mag)
	}
	atten := -20 * math.Log10(worst)
	if math.IsNaN(ripple) || math.IsInf(ripple, 0) ||
		math.IsNaN(atten) || math.IsInf(atten, 0) {
		return nil, fmt.Errorf("resample: analysis produced non-finite results")
	}

	return &QualityReport{
		PassbandRippleDB:      ripple,
		StopbandAttenuationDB: atten,
		Taps:                  r.taps,
		Delay:                 int(r.shift),
	}, nil
}

// validate checks that the two bands are finite, ordered, non-empty,
// non-overlapping and within [0, domainHi], where domainHi is the higher of
// the two Nyquist frequencies.
func (b Bands) validate(domainHi float64) error {
	edges := []struct {
		name string
		v    float64
	}{
		{"passband low", b.PassbandLow},
		{"passband high", b.PassbandHigh},
		{"stopband low", b.StopbandLow},
		{"stopband high", b.StopbandHigh},
	}
	for _, e := range edges {
		if math.IsNaN(e.v) || math.IsInf(e.v, 0) {
			return fmt.Errorf("resample: %s frequency must be finite, got %g", e.name, e.v)
		}
		if e.v < 0 || e.v > domainHi {
			return fmt.Errorf("resample: %s frequency %g Hz is outside [0, %g] Nyquist",
				e.name, e.v, domainHi)
		}
	}
	if !(b.PassbandLow < b.PassbandHigh) {
		return fmt.Errorf("resample: passband %g..%g Hz is empty or inverted",
			b.PassbandLow, b.PassbandHigh)
	}
	if !(b.StopbandLow < b.StopbandHigh) {
		return fmt.Errorf("resample: stopband %g..%g Hz is empty or inverted",
			b.StopbandLow, b.StopbandHigh)
	}
	if b.PassbandHigh > b.StopbandLow {
		return fmt.Errorf("resample: passband %g..%g Hz overlaps stopband %g..%g Hz",
			b.PassbandLow, b.PassbandHigh, b.StopbandLow, b.StopbandHigh)
	}
	return nil
}
