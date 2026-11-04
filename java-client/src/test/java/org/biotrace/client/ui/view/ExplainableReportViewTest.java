package org.biotrace.client.ui.view;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.*;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ExplainableReportView UI Tests")
class ExplainableReportViewTest {

    @Test
    @DisplayName("Verify ExplainableReportView instantiates components and sets job context cleanly")
    void testViewComponentsAndJobContext() {
        if (GraphicsEnvironment.isHeadless()) {
            return;
        }

        ClientConfig config = ClientConfig.loadFromEnvironment();
        HttpForensicApiClient apiClient = new HttpForensicApiClient(config);
        EventBus eventBus = new EventBus();

        ExplainableReportView view = new ExplainableReportView(apiClient, eventBus);
        assertNotNull(view);

        // Verify context setting does not crash
        assertDoesNotThrow(() -> view.setJobId("job-test-12345"));
    }
}
