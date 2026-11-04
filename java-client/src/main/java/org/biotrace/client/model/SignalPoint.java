package org.biotrace.client.model;

/**
 * A discrete physiological or dynamic sample point in time.
 */
public record SignalPoint(
        double timestampSeconds,
        long frameIndex,
        double value,
        double confidence
) implements Comparable<SignalPoint> {

    @Override
    public int compareTo(SignalPoint o) {
        return Double.compare(this.timestampSeconds, o.timestampSeconds);
    }
}
