package org.biotrace.client.service.repository;

import java.util.List;
import java.util.Optional;

/**
 * Generic repository interface demonstrating Java Generics, Abstraction, and Collections.
 *
 * @param <T>  Entity type.
 * @param <ID> Primary key identifier type.
 */
public interface Repository<T, ID> {

    /**
     * Persists or updates the entity.
     */
    T save(T entity);

    /**
     * Finds entity by unique identifier.
     */
    Optional<T> findById(ID id);

    /**
     * Retrieves all persisted entities.
     */
    List<T> findAll();

    /**
     * Deletes entity by unique identifier.
     */
    boolean deleteById(ID id);

    /**
     * Returns total count of entities.
     */
    long count();
}
