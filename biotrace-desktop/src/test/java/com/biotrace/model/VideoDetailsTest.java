package com.biotrace.model;

import org.junit.jupiter.api.Test;
import java.io.File;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VideoDetails model.
 * Verifies that setters/getters work and that defaults are safe.
 */
public class VideoDetailsTest {

    private static File tempFile() throws Exception {
        File f = File.createTempFile("biotrace_test_", ".mp4");
        f.deleteOnExit();
        return f;
    }

    @Test
    void defaultsAreSafe() throws Exception {
        VideoDetails d = new VideoDetails(tempFile());
        assertEquals(0,         d.getWidth());
        assertEquals(0,         d.getHeight());
        assertEquals(0.0,       d.getFps(), 1e-9);
        assertEquals(-1L,       d.getFrameCount());
        assertEquals("Unknown", d.getDuration());
        assertEquals("Unknown", d.getResolution());
    }

    @Test
    void settersUpdateGetters() throws Exception {
        VideoDetails d = new VideoDetails(tempFile());
        d.setWidth(1920);
        d.setHeight(1080);
        d.setFps(29.97);
        d.setFrameCount(5000L);
        d.setDuration("2:47");

        assertEquals(1920,    d.getWidth());
        assertEquals(1080,    d.getHeight());
        assertEquals(29.97,   d.getFps(), 1e-6);
        assertEquals(5000L,   d.getFrameCount());
        assertEquals("2:47",  d.getDuration());
        assertEquals("1920x1080", d.getResolution());
    }

    @Test
    void fileSizeMatchesActualFile() throws Exception {
        File f = tempFile();
        VideoDetails d = new VideoDetails(f);
        assertEquals(f.length(), d.getSizeBytes());
    }

    @Test
    void filenameMatchesFileName() throws Exception {
        File f = tempFile();
        VideoDetails d = new VideoDetails(f);
        assertEquals(f.getName(), d.getFilename());
    }
}
