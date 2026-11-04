package org.biotrace.client.ui.theme;

import com.formdev.flatlaf.FlatLightLaf;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;

/**
 * Visual design system for BIOTrace:
 * Professional, modern, research-grade scientific workstation styling.
 * Employs a light scientific workstation theme with warm white/light grey canvas,
 * crisp white elevated cards, charcoal typography, deep teal primary accents,
 * restrained physiological green, warning amber, and critical alert red.
 */
public final class ForensicTheme {

    private ForensicTheme() {}

    // --- Core Canvas & Surface Palette (Light Scientific Workstation) ---
    public static final Color BG_CANVAS = new Color(0xF8, 0xF9, 0xFA);        // Main workstation canvas (warm laboratory grey)
    public static final Color BG_SURFACE = new Color(0xFF, 0xFF, 0xFF);       // Elevated card surface (pure crisp white)
    public static final Color BG_PANEL = new Color(0xF1, 0xF5, 0xF9);         // Sub-containers & secondary panel backgrounds
    public static final Color BG_HOVER = new Color(0xEA, 0xEE, 0xF4);         // Component hover / alternating table row
    public static final Color BG_ACTIVE = new Color(0xE0, 0xF2, 0xFE);        // Selection / active state highlight

    // Structural Borders
    public static final Color BORDER_SUBTLE = new Color(0xE2, 0xE8, 0xF0);    // Subtle hairline structural divider
    public static final Color BORDER_STRONG = new Color(0xCB, 0xD5, 0xE1);    // Elevated perimeter outline
    public static final Color BORDER_FOCUS = new Color(0x0F, 0x76, 0x6E);     // Active focus ring

    // Primary Accents (Scientific Deep Teal & Intelligence Blue)
    public static final Color TEAL_PRIMARY = new Color(0x0F, 0x76, 0x6E);     // Deep authoritative forensic teal
    public static final Color TEAL_HOVER = new Color(0x11, 0x5E, 0x59);       // Teal interactive hover
    public static final Color TEAL_LIGHT = new Color(0xCC, 0xFB, 0xF1);       // Soft teal background tint
    public static final Color TEAL_BORDER = new Color(0x99, 0xF6, 0xE4);      // Soft teal badge border
    public static final Color TEAL_ACCENT = new Color(0x0D, 0x94, 0x88);      // Vibrant telemetry teal

    public static final Color CYAN_ACCENT = new Color(0x02, 0x84, 0xC7);      // Telemetry / active signal cyan-blue
    public static final Color CYAN_LIGHT = new Color(0xF0, 0xF9, 0xFF);       // Soft cyan background tint
    public static final Color CYAN_BORDER = new Color(0xBA, 0xE6, 0xFD);      // Soft cyan border

    // Restrained Forensic Verdict & Status Accents
    public static final Color SUCCESS_GREEN = new Color(0x16, 0xA3, 0x4A);    // Physiological consistency (authentic)
    public static final Color SUCCESS_BG = new Color(0xDC, 0xFC, 0xE7);       // Light green badge background
    public static final Color SUCCESS_BORDER = new Color(0x86, 0xEF, 0xAC);   // Light green badge border

    public static final Color ALERT_RED = new Color(0xDC, 0x26, 0x26);        // Anomaly / manipulation detected (fake)
    public static final Color ALERT_BG = new Color(0xFE, 0xE2, 0xE2);         // Light red badge background
    public static final Color ALERT_BORDER = new Color(0xFC, 0xA5, 0xA5);     // Light red badge border

    public static final Color WARNING_AMBER = new Color(0xD9, 0x77, 0x06);    // Indeterminate / pending / cautionary
    public static final Color WARNING_BG = new Color(0xFE, 0xF3, 0xC7);       // Light amber badge background
    public static final Color WARNING_BORDER = new Color(0xFC, 0xD3, 0x4D);   // Light amber badge border

    public static final Color NEUTRAL_SLATE = new Color(0x64, 0x74, 0x8B);    // Inconclusive / cancelled / inactive
    public static final Color NEUTRAL_BG = new Color(0xF1, 0xF5, 0xF9);       // Light slate badge background
    public static final Color NEUTRAL_BORDER = new Color(0xCB, 0xD5, 0xE1);   // Light slate badge border

    // Charcoal Typography Hierarchy
    public static final Color TEXT_PRIMARY = new Color(0x0F, 0x17, 0x2A);     // Slate-900 high-contrast charcoal
    public static final Color TEXT_SECONDARY = new Color(0x33, 0x41, 0x55);   // Slate-700 body & label text
    public static final Color TEXT_MUTED = new Color(0x64, 0x74, 0x8B);       // Slate-500 secondary captions & hints
    public static final Color TEXT_DIM = new Color(0x94, 0xA3, 0xB8);         // Slate-400 subtle metadata & timestamps

    // --- Backward Compatibility Aliases ---
    public static final Color BG_DARK = BG_CANVAS;
    public static final Color BG_CARD = BG_SURFACE;
    public static final Color BG_CARD_HOVER = BG_HOVER;
    public static final Color BORDER = BORDER_SUBTLE;
    public static final Color EMERALD = SUCCESS_GREEN;
    public static final Color CYAN = TEAL_PRIMARY;

    // --- Typography Fonts ---
    public static final Font FONT_TITLE_LARGE = new Font("Segoe UI", Font.BOLD, 20);
    public static final Font FONT_TITLE = new Font("Segoe UI", Font.BOLD, 16);
    public static final Font FONT_SUBTITLE = new Font("Segoe UI", Font.BOLD, 13);
    public static final Font FONT_BODY = new Font("Segoe UI", Font.PLAIN, 13);
    public static final Font FONT_BODY_BOLD = new Font("Segoe UI", Font.BOLD, 13);
    public static final Font FONT_CODE = new Font("Consolas", Font.PLAIN, 12);
    public static final Font FONT_CODE_BOLD = new Font("Consolas", Font.BOLD, 12);
    public static final Font FONT_BADGE = new Font("Segoe UI", Font.BOLD, 11);
    public static final Font FONT_CAPTION = new Font("Segoe UI", Font.PLAIN, 11);

    /**
     * Initializes FlatLaf Light theme with refined scientific workstation UI defaults.
     */
    public static void applyTheme() {
        try {
            FlatLightLaf.setup();

            // Core Global Defaults
            UIManager.put("Panel.background", BG_CANVAS);
            UIManager.put("Label.foreground", TEXT_PRIMARY);
            UIManager.put("Label.font", FONT_BODY);
            
            // Rounded corners & clean geometry
            UIManager.put("Button.arc", 8);
            UIManager.put("Component.arc", 8);
            UIManager.put("ProgressBar.arc", 8);
            UIManager.put("ScrollBar.thumbArc", 8);
            UIManager.put("ScrollBar.width", 10);
            
            // Text fields & controls
            UIManager.put("TextField.background", BG_SURFACE);
            UIManager.put("TextField.foreground", TEXT_PRIMARY);
            UIManager.put("TextField.borderColor", BORDER_SUBTLE);
            UIManager.put("TextField.focusedBorderColor", TEAL_PRIMARY);
            UIManager.put("ComboBox.background", BG_SURFACE);
            UIManager.put("ComboBox.foreground", TEXT_PRIMARY);

            // High-grade scientific tables
            UIManager.put("Table.background", BG_SURFACE);
            UIManager.put("Table.foreground", TEXT_PRIMARY);
            UIManager.put("Table.gridColor", BORDER_SUBTLE);
            UIManager.put("Table.selectionBackground", BG_ACTIVE);
            UIManager.put("Table.selectionForeground", TEXT_PRIMARY);
            UIManager.put("Table.rowHeight", 32);
            UIManager.put("TableHeader.background", BG_PANEL);
            UIManager.put("TableHeader.foreground", TEXT_PRIMARY);
            UIManager.put("TableHeader.font", FONT_BODY_BOLD);
            UIManager.put("TableHeader.separatorColor", BORDER_SUBTLE);

            // Scrollbars
            UIManager.put("ScrollPane.borderColor", BORDER_SUBTLE);
            UIManager.put("ScrollBar.track", BG_CANVAS);
            UIManager.put("ScrollBar.thumb", BORDER_STRONG);

            // Tabs & Separators
            UIManager.put("Separator.foreground", BORDER_SUBTLE);
            UIManager.put("TabbedPane.selectedBackground", BG_SURFACE);
            UIManager.put("TabbedPane.selectedForeground", TEAL_PRIMARY);
            UIManager.put("TabbedPane.underlineColor", TEAL_PRIMARY);
        } catch (Exception e) {
            System.err.println("Could not apply FlatLightLaf theme: " + e.getMessage());
        }
    }

    public static Border createCardBorder() {
        return BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(14, 18, 14, 18)
        );
    }

    public static Border createSubtleBorder() {
        return BorderFactory.createLineBorder(BORDER_SUBTLE, 1);
    }

    public static Border createPanelPadding(int top, int left, int bottom, int right) {
        return BorderFactory.createEmptyBorder(top, left, bottom, right);
    }
}
