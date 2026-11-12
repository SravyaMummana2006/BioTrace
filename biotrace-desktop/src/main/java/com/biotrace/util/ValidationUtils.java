package com.biotrace.util;

import java.io.File;

public class ValidationUtils {
    public static boolean isValidVideoFile(File file) {
        if (file == null || !file.exists() || !file.isFile() || !file.canRead()) {
            return false;
        }
        String name = file.getName().toLowerCase();
        return name.endsWith(".mp4") || name.endsWith(".avi") || name.endsWith(".mkv") || name.endsWith(".mov");
    }
}
