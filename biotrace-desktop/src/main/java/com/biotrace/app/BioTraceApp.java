package com.biotrace.app;

import com.biotrace.ui.MainFrame;
import com.biotrace.util.OpenCVUtils;
import com.formdev.flatlaf.FlatDarkLaf;

import javax.swing.SwingUtilities;

public class BioTraceApp {
    public static void main(String[] args) {
        OpenCVUtils.loadLibrary();
        
        FlatDarkLaf.setup();
        
        SwingUtilities.invokeLater(() -> {
            MainFrame frame = new MainFrame();
            frame.setVisible(true);
        });
    }
}
