package org.biotrace.client.model;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Continuous 1D physiological or biometric signal trace (e.g., rPPG BVP, EAR blink contour).
 * Demonstrates defensive copying and Java Stream statistical operations.
 */
public record ForensicSignal(
        String modality,
        String signalName,
        double samplingRateHz,
        List<SignalPoint> points,
        Map<String, String> metadata
) {
    public ForensicSignal {
        points = (points != null) ? List.copyOf(points) : List.of();
        metadata = (metadata != null) ? Map.copyOf(metadata) : Map.of();
    }

    public double getMeanValue() {
        return points.stream()
                .mapToDouble(SignalPoint::value)
                .average()
                .orElse(0.0);
    }

    public double getMinValue() {
        return points.stream()
                .mapToDouble(SignalPoint::value)
                .min()
                .orElse(0.0);
    }

    public double getMaxValue() {
        return points.stream()
                .mapToDouble(SignalPoint::value)
                .max()
                .orElse(0.0);
    }

    public int size() {
        return points.size();
    }
}
