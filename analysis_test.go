// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package resample

import (
	"math"
	"testing"
)

// TestAnalyzeReport checks the headline figures against the design targets and
// the Resampler accessors. The measured stopband attenuation must clear the
// requested attenuation, and the cost and delay reported must be exactly the
// values the built resampler exposes.
func TestAnalyzeReport(t *testing.T) {
	presets := []struct {
		name string
		q    Quality
	}{
		{"Fast", Fast}, {"Balanced", Balanced},
		{"HighQuality", HighQuality}, {"Archival", Archival},
	}
	cases := []struct{ in, out float64 }{
		{44100, 48000}, {48000, 44100}, {96000, 48000}, {48000, 96000},
	}

	for _, p := range presets {
		for _, c := range cases {
			t.Run(p.name+"/"+rateName(c.in, c.out), func(t *testing.T) {
				rep, err := Analyze(c.in, c.out, p.q, Bands{})
				if err != nil {
					t.Fatal(err)
				}
				r, err := New(c.in, c.out, p.q)
				if err != nil {
					t.Fatal(err)
				}

				if rep.Taps != r.Taps() {
					t.Errorf("taps %d != Resampler.Taps %d", rep.Taps, r.Taps())
				}
				if rep.Delay != r.Delay() {
					t.Errorf("delay %d != Resampler.Delay %d", rep.Delay, r.Delay())
				}
				if rep.StopbandAttenuationDB < p.q.Attenuation {
					t.Errorf("stopband attenuation %.2f dB below the design target %.0f dB",
						rep.StopbandAttenuationDB, p.q.Attenuation)
				}
				if rep.PassbandRippleDB < 0 {
					t.Errorf("negative ripple %g", rep.PassbandRippleDB)
				}
				if rep.PassbandRippleDB > 1 {
					t.Errorf("passband ripple %.4f dB exceeds 1 dB", rep.PassbandRippleDB)
				}
				if math.IsNaN(rep.PassbandRippleDB) || math.IsInf(rep.PassbandRippleDB, 0) ||
					math.IsNaN(rep.StopbandAttenuationDB) || math.IsInf(rep.StopbandAttenuationDB, 0) {
					t.Fatal("non-finite report field")
				}
				t.Logf("%-12s %6.0f->%-6.0f ripple %8.4f dB  atten %9.3f dB  taps %5d delay %d",
					p.name, c.in, c.out, rep.PassbandRippleDB,
					rep.StopbandAttenuationDB, rep.Taps, rep.Delay)
			})
		}
	}
}

// TestAnalyzeDeterminism pins the determinism promise: the same inputs must
// produce byte-identical reports, and an explicit band must give a stable
// answer different from the default one.
func TestAnalyzeDeterminism(t *testing.T) {
	const in, out = 44100.0, 48000.0
	first, err := Analyze(in, out, HighQuality, Bands{})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		got, err := Analyze(in, out, HighQuality, Bands{})
		if err != nil {
			t.Fatal(err)
		}
		if *got != *first {
			t.Fatalf("run %d differs: %+v vs %+v", i, got, first)
		}
	}

	explicit, err := Analyze(in, out, HighQuality, Bands{
		PassbandLow: 100, PassbandHigh: 15000,
		StopbandLow: 23000, StopbandHigh: 24000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if explicit.StopbandAttenuationDB == first.StopbandAttenuationDB ||
		explicit.Taps != first.Taps {
		t.Fatal("explicit bands did not change the response measurement")
	}
}

// TestAnalyzeMatchesActualFilter independently evaluates the response of the
// phases New built by direct summation and compares it with the FFT-based
// report. The polyphase identity |H_p(e^jw)| == |H_proto(e^jw/l)|/l means the
// two routes must agree apart from the FFT grid's sampling error.
func TestAnalyzeMatchesActualFilter(t *testing.T) {
	const in, out = 44100.0, 48000.0
	r, err := New(in, out, HighQuality)
	if err != nil {
		t.Fatal(err)
	}
	proto := make([]float64, r.taps*r.l)
	for p, ph := range r.phases {
		for j, v := range ph {
			proto[p+j*r.l] = v
		}
	}
	nfft := 1
	for nfft < 2*len(proto) {
		nfft <<= 1
	}
	re := make([]float64, nfft)
	im := make([]float64, nfft)
	copy(re, proto)
	fft(re, im)

	maxErr := 0.0
	for _, p := range []int{0, 1, 7, 33, r.l - 1} {
		ph := r.phases[p]
		for _, f := range []float64{100, 1000, 5000, 15000, 20000} {
			omega := 2 * math.Pi * f / in
			var dr, di float64
			for j, v := range ph {
				dr += v * math.Cos(omega*float64(j))
				di -= v * math.Sin(omega*float64(j))
			}
			direct := math.Hypot(dr, di)

			k := int(math.Round(float64(nfft) * f / (float64(r.l) * in)))
			fromFFT := math.Hypot(re[k], im[k]) / float64(r.l)
			maxErr = math.Max(maxErr, math.Abs(direct-fromFFT)/direct)
		}
	}
	if maxErr > 1e-6 {
		t.Fatalf("FFT route disagrees with direct phase response by %.3e", maxErr)
	}
}

// TestAnalyzeErrors exercises every documented failure mode: illegal rates
// and qualities, bands outside Nyquist, empty or inverted bands, overlapping
// bands, non-finite edges, and ratios whose filter would exceed the analysis
// budget.
func TestAnalyzeErrors(t *testing.T) {
	const in, out = 48000.0, 96000.0
	valid := Bands{
		PassbandLow: 100, PassbandHigh: 18000,
		StopbandLow: 26000, StopbandHigh: 47000,
	}

	t.Run("rates", func(t *testing.T) {
		for _, c := range []struct{ in, out float64 }{
			{0, out}, {-1, out}, {math.NaN(), out},
			{math.Inf(1), out}, {in, 0}, {in, math.Inf(-1)},
		} {
			if _, err := Analyze(c.in, c.out, HighQuality, valid); err == nil {
				t.Errorf("Analyze(%g, %g) expected error", c.in, c.out)
			}
		}
	})

	t.Run("quality", func(t *testing.T) {
		for _, q := range []Quality{
			{Attenuation: 10, Transition: 0.05},
			{Attenuation: 120, Transition: 0},
			{Attenuation: 120, Transition: 1},
			{Attenuation: math.NaN(), Transition: 0.05},
		} {
			if _, err := Analyze(in, out, q, valid); err == nil {
				t.Errorf("Analyze with quality %+v expected error", q)
			}
		}
	})

	t.Run("bands", func(t *testing.T) {
		domainHi := math.Max(in, out) / 2
		for _, b := range []Bands{
			{100, 100, 26000, 47000},          // empty passband
			{100, 18000, 26000, 26000},        // empty stopband
			{18000, 100, 26000, 47000},        // inverted passband
			{100, 18000, 47000, 26000},        // inverted stopband
			{100, 30000, 26000, 47000},        // overlapping bands
			{100, 18000, 26000, domainHi + 1}, // above higher Nyquist
			{-1, 18000, 26000, 47000},         // negative edge
			{100, math.NaN(), 26000, 47000},   // non-finite edge
			{100, 18000, math.Inf(1), 47000},  // infinite edge
		} {
			if _, err := Analyze(in, out, HighQuality, b); err == nil {
				t.Errorf("Analyze with bands %+v expected error", b)
			}
		}
	})

	t.Run("budget", func(t *testing.T) {
		// A ratio of one to a million with an aggressive transition needs a
		// prototype far beyond maxAnalysisFFT; it must fail clearly rather
		// than attempt the allocation.
		if _, err := Analyze(1, 1e6, HighQuality, Bands{}); err == nil {
			t.Error("extreme ratio expected budget error")
		}
	})

	t.Run("default_bands_no_room", func(t *testing.T) {
		// Equal or nearly equal rates leave no measurable interval above the
		// lower Nyquist, so the default bands cannot be derived.
		for _, c := range []struct{ in, out float64 }{
			{48000, 48000}, {48000, 47999},
		} {
			if _, err := Analyze(c.in, c.out, Fast, Bands{}); err == nil {
				t.Errorf("Analyze(%g, %g) with default bands expected error", c.in, c.out)
			}
		}
		// Explicit bands still work: the filter exists and can be measured
		// over any valid intervals, and band validation still applies.
		if _, err := Analyze(48000, 48000, Fast, Bands{
			PassbandLow: 0, PassbandHigh: 18000,
			StopbandLow: 26000, StopbandHigh: 24000 - 1,
		}); err == nil {
			t.Error("expected error for inverted explicit bands")
		}
		if _, err := Analyze(48000, 48000, Fast, Bands{
			PassbandLow: 0, PassbandHigh: 15000,
			StopbandLow: 20000, StopbandHigh: 24000,
		}); err != nil {
			t.Errorf("explicit bands at equal rates failed: %v", err)
		}
	})
}

// coherentAmplitude measures one spectral component's amplitude with a Hann
// windowed single-bin DFT. The window removes the need for the segment to
// contain a whole number of cycles; for a unit sine the result is one.
func coherentAmplitude(x []float64, freqHz, rate float64) float64 {
	w := 2 * math.Pi * freqHz / rate
	var sr, si, wsum float64
	for i, v := range x {
		hann := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(len(x)-1)))
		vw := v * hann
		wsum += hann
		sr += vw * math.Cos(w*float64(i))
		si -= vw * math.Sin(w*float64(i))
	}
	return 2 * math.Hypot(sr, si) / wsum
}

// TestAnalyzeSineSweepCrossCheck is the independent end-to-end check: fixed
// passband and stopband sine sweeps are pushed through the real Write/Flush
// pipeline, and their measured gain and residual are compared in order of
// magnitude with the report derived purely from the filter coefficients.
func TestAnalyzeSineSweepCrossCheck(t *testing.T) {
	const n = 1 << 15
	const inRate, outRate = 96000.0, 48000.0

	passSweep := []float64{200, 1000, 4000, 8000, 14000, 19000}
	stopSweep := []float64{26000, 30000, 36000, 42000, 46000}
	bands := Bands{
		PassbandLow: 100, PassbandHigh: 19500,
		StopbandLow: 25000, StopbandHigh: 47000,
	}
	presets := []struct {
		name string
		q    Quality
	}{
		{"Fast", Fast}, {"Balanced", Balanced},
		{"HighQuality", HighQuality}, {"Archival", Archival},
	}

	prevAtten := math.Inf(-1)
	prevRipple := math.Inf(1)
	for _, p := range presets {
		rep, err := Analyze(inRate, outRate, p.q, bands)
		if err != nil {
			t.Fatal(err)
		}

		// Passband: coherent amplitude of every swept tone must stay within
		// a small margin of the report's ripple figure.
		measuredRipple := 0.0
		for _, f := range passSweep {
			out, err := Resample(tone(n, f, inRate), inRate, outRate, p.q)
			if err != nil {
				t.Fatal(err)
			}
			mid := out[len(out)/4 : 3*len(out)/4]
			amp := coherentAmplitude(mid, f, outRate)
			measuredRipple = math.Max(measuredRipple, math.Abs(20*math.Log10(amp)))
		}

		// Stopband: a tone above the output Nyquist folds into the band; the
		// residual peak over the swept tones is its measured rejection.
		measuredResidual := -400.0
		for _, f := range stopSweep {
			out, err := Resample(tone(n, f, inRate), inRate, outRate, p.q)
			if err != nil {
				t.Fatal(err)
			}
			mid := out[len(out)/4 : 3*len(out)/4]
			peak := 0.0
			for _, v := range mid {
				peak = math.Max(peak, math.Abs(v))
			}
			measuredResidual = math.Max(measuredResidual, 20*math.Log10(peak))
		}
		measuredAtten := -measuredResidual

		t.Logf("%-12s report ripple %8.5f / measured %8.5f dB; report atten %9.3f / measured %9.3f dB",
			p.name, rep.PassbandRippleDB, measuredRipple,
			rep.StopbandAttenuationDB, measuredAtten)

		if measuredRipple > rep.PassbandRippleDB+0.01 {
			t.Errorf("%s: measured ripple %.5f dB exceeds report %.5f dB by more than 0.01 dB",
				p.name, measuredRipple, rep.PassbandRippleDB)
		}
		// The report is the worst point of the whole stopband grid; a finite
		// set of swept tones can only match or beat it, within a generous
		// margin for measurement and aliasing differences.
		if measuredAtten < rep.StopbandAttenuationDB-25 {
			t.Errorf("%s: measured attenuation %.2f dB is more than 25 dB below the report %.2f dB",
				p.name, measuredAtten, rep.StopbandAttenuationDB)
		}
		if measuredAtten <= 0 {
			t.Errorf("%s: non-positive measured attenuation", p.name)
		}

		// Preset ordering: stronger designs must measure strictly better on
		// both figures.
		if !(rep.StopbandAttenuationDB > prevAtten) {
			t.Errorf("stopband ordering broken at %s: %.2f after %.2f",
				p.name, rep.StopbandAttenuationDB, prevAtten)
		}
		if !(rep.PassbandRippleDB <= prevRipple) {
			t.Errorf("passband ordering broken at %s: %.5f after %.5f",
				p.name, rep.PassbandRippleDB, prevRipple)
		}
		prevAtten = rep.StopbandAttenuationDB
		prevRipple = rep.PassbandRippleDB
	}
}
