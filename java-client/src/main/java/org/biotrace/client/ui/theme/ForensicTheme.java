package org.biotrace.client.ui.theme;

import com.formdev.flatlaf.FlatDarkLaf;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;

/**
 * Visual design system for BIOTrace:
 * Scientific, dark forensic styling with muted emerald and cyan accents.
 */
public final class ForensicTheme {

    private ForensicTheme() {}

    // --- Color Palette ---
    public static final Color BG_DARK = new Color(0x10, 0x14, 0x18);          // Deep Charcoal Canvas
    public static final Color BG_CARD = new Color(0x17, 0x1D, 0x24);          // Elevated Surface
    public static final Color BG_CARD_HOVER = new Color(0x20, 0x27, 0x31);    // Hover state
    public static final Color BORDER = new Color(0x27, 0x32, 0x3F);           // Subtle structural borders
    
    // Accents
    public static final Color EMERALD = new Color(0x10, 0xB9, 0x81);         // Physiological Consistency / Real
    public static final Color CYAN = new Color(0x06, 0xB6, 0xD4);            // Forensic Intelligence / Active
    public static final Color ALERT_RED = new Color(0xEF, 0x44, 0x44);       // Anomaly / Inconsistency / Fake
    public static final Color WARNING_AMBER = new Color(0xF5, 0x9E, 0x0B);   // Indeterminate / Warning
    
    // Typography Colors
    public static final Color TEXT_PRIMARY = new Color(0xF1, 0xF5, 0xF9);     // High contrast white
    public static final Color TEXT_MUTED = new Color(0x94, 0xA3, 0xB8);       // Secondary muted slate
    public static final Color TEXT_DIM = new Color(0x64, 0x74, 0x8B);         // Timestamps / subtle metadata

    // --- Typography Fonts ---
    public static final Font FONT_TITLE = new Font("Segoe UI", Font.BOLD, 18);
    public static final Font FONT_SUBTITLE = new Font("Segoe UI", Font.BOLD, 14);
    public static final Font FONT_BODY = new Font("Segoe UI", Font.PLAIN, 13);
    public static final Font FONT_CODE = new Font("Consolas", Font.PLAIN, 12);
    public static final Font FONT_BADGE = new Font("Segoe UI", Font.BOLD, 11);

    /**
     * Initializes FlatLaf Dark theme with custom scientific UI defaults.
     */
    public static void applyTheme() {
        try {
            FlatDarkLaf.setup();

            UIManager.put("Panel.background", BG_DARK);
            UIManager.put("Label.foreground", TEXT_PRIMARY);
            UIManager.put("Label.font", FONT_BODY);
            UIManager.put("Button.arc", 6);
            UIManager.put("Component.arc", 6);
            UIManager.put("ProgressBar.arc", 6);
            UIManager.put("ScrollBar.thumbArc", 6);
            UIManager.put("ScrollBar.width", 10);
            UIManager.put("Table.background", BG_CARD);
            UIManager.put("Table.foreground", TEXT_PRIMARY);
            UIManager.put("Table.gridColor", BORDER);
            UIManager.put("TableHeader.background", BG_CARD_HOVER);
            UIManager.put("TableHeader.foreground", TEXT_PRIMARY);
        } catch (Exception e) {
            System.err.println("Could not apply FlatDarkLaf theme: " + e.getMessage());
        }
    }

    public static Border createCardBorder() {
        return BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(BORDER, 1),
                BorderFactory.createEmptyBorder(12, 14, 12, 14)
        );
    }
}
