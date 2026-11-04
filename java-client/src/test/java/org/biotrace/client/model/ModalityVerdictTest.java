package org.biotrace.client.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ModalityVerdict Source and Binary Compatibility Tests")
class ModalityVerdictTest {

    @Test
    @DisplayName("Backward-compatible constructor with primitive double should populate Optional.of")
    void testOverloadedDoubleConstructorCompatibility() {
        // Simulates code calling the legacy 6-argument constructor with primitive double
        ModalityVerdict verdict = new ModalityVerdict("blink", 0.95, false, 0.99, Optional.empty(), "High consistency");

        assertTrue(verdict.authenticityScore().isPresent());
        assertEquals(0.95, verdict.authenticityScore().get(), 1e-6);
        assertTrue(verdict.isAuthenticityAssessed());
        assertEquals(0.95, verdict.getAuthenticityScoreOrDefault(0.0), 1e-6);
    }

    @Test
    @DisplayName("Physiological-only 5-argument constructor should populate Optional.empty")
    void testPhysiologicalOnlyConstructor() {
        ModalityVerdict verdict = new ModalityVerdict("rppg", false, 0.85, Optional.empty(), "Pulse 72 BPM");

        assertTrue(verdict.authenticityScore().isEmpty());
        assertFalse(verdict.isAuthenticityAssessed());
        assertEquals(Double.NaN, verdict.getAuthenticityScoreOrDefault(Double.NaN));
    }

    @Test
    @DisplayName("Canonical constructor with null Optional arguments must normalize safely")
    void testDefensiveNullNormalization() {
        ModalityVerdict verdict = new ModalityVerdict("rppg", null, false, 0.80, null, "Null arguments handled");

        assertNotNull(verdict.authenticityScore(), "authenticityScore must normalize to non-null Optional");
        assertTrue(verdict.authenticityScore().isEmpty());
        assertNotNull(verdict.signalTrace(), "signalTrace must normalize to non-null Optional");
        assertTrue(verdict.signalTrace().isEmpty());
        assertFalse(verdict.isAuthenticityAssessed());
    }

    @Test
    @DisplayName("Range validations must enforce [0.0, 1.0] for present score and confidence")
    void testRangeValidation() {
        assertThrows(IllegalArgumentException.class, () ->
                new ModalityVerdict("rppg", Optional.of(1.2), false, 0.5, Optional.empty(), "Invalid score"));

        assertThrows(IllegalArgumentException.class, () ->
                new ModalityVerdict("rppg", Optional.of(-0.1), false, 0.5, Optional.empty(), "Invalid score"));

        assertThrows(IllegalArgumentException.class, () ->
                new ModalityVerdict("rppg", Optional.empty(), false, -0.01, Optional.empty(), "Invalid confidence"));

        assertThrows(IllegalArgumentException.class, () ->
                new ModalityVerdict("rppg", Optional.empty(), false, 1.05, Optional.empty(), "Invalid confidence"));
    }
}
