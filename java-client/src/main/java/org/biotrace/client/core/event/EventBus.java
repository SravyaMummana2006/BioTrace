package org.biotrace.client.core.event;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Thread-safe EventBus for decoupling UI views from background services.
 * Demonstrates Generics, Functional Interfaces, and Concurrent Collections.
 */
public class EventBus {

    private final Map<Class<?>, List<Consumer<?>>> subscribers = new ConcurrentHashMap<>();

    /**
     * Subscribes a listener to events of type T.
     */
    public <T> void subscribe(Class<T> eventType, Consumer<T> listener) {
        subscribers.computeIfAbsent(eventType, k -> new CopyOnWriteArrayList<>())
                   .add(listener);
    }

    /**
     * Publishes an event to all registered subscribers of that type.
     */
    @SuppressWarnings("unchecked")
    public <T> void publish(T event) {
        if (event == null) return;
        List<Consumer<?>> listeners = subscribers.get(event.getClass());
        if (listeners != null) {
            for (Consumer<?> listener : listeners) {
                ((Consumer<T>) listener).accept(event);
            }
        }
    }

    /**
     * Clears all subscribers (useful for testing).
     */
    public void clear() {
        subscribers.clear();
    }
}
