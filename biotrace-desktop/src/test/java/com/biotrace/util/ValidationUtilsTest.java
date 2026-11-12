package com.biotrace.util;

import com.biotrace.model.VideoDetails;
import org.junit.jupiter.api.Test;
import java.io.File;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ValidationUtils.
 * No OpenCV native library or real video file is required.
 */
public class ValidationUtilsTest {

    @Test
    void nullFileReturnsFalse() {
        assertFalse(ValidationUtils.isValidVideoFile(null),
                "null input must return false");
    }

    @Test
    void nonExistentFileReturnsFalse() {
        File ghost = new File("does_not_exist.mp4");
        assertFalse(ValidationUtils.isValidVideoFile(ghost),
                "non-existent file must return false even with valid extension");
    }

    @Test
    void directoryReturnsFalse() {
        // System temp dir always exists and is a directory, not a file
        File dir = new File(System.getProperty("java.io.tmpdir"));
        assertFalse(ValidationUtils.isValidVideoFile(dir),
                "a directory must not be accepted as a video file");
    }

    @Test
    void unsupportedExtensionReturnsFalse() throws Exception {
        // Create a real, readable temp file with a wrong extension
        File tmp = File.createTempFile("test_video_", ".txt");
        tmp.deleteOnExit();
        try {
            assertFalse(ValidationUtils.isValidVideoFile(tmp),
                    ".txt extension must be rejected");
        } finally {
            tmp.delete();
        }
    }

    @Test
    void allSupportedExtensionsAreRecognised() throws Exception {
        for (String ext : new String[]{ ".mp4", ".avi", ".mkv", ".mov" }) {
            File tmp = File.createTempFile("test_video_", ext);
            tmp.deleteOnExit();
            try {
                // The file exists and is readable; extension is valid.
                // isValidVideoFile checks existence + readability + extension only.
                assertTrue(ValidationUtils.isValidVideoFile(tmp),
                        "extension " + ext + " should be accepted when file exists");
            } finally {
                tmp.delete();
            }
        }
    }

    @Test
    void extensionCheckIsCaseInsensitive() throws Exception {
        for (String ext : new String[]{ ".MP4", ".AVI", ".MKV", ".MOV" }) {
            File tmp = File.createTempFile("test_video_", ext);
            tmp.deleteOnExit();
            try {
                assertTrue(ValidationUtils.isValidVideoFile(tmp),
                        "uppercase extension " + ext + " should be accepted");
            } finally {
                tmp.delete();
            }
        }
    }
}
