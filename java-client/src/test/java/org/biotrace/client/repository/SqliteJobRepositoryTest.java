package org.biotrace.client.repository;

import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.model.ProcessingStage;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SqliteJobRepository CRUD and Query Tests")
class SqliteJobRepositoryTest {

    private SqliteJobRepository repository;

    @BeforeEach
    void setUp() {
        repository = SqliteJobRepository.createInMemory();
    }

    @Test
    @DisplayName("Should save and find job by ID")
    void testSaveAndFind() {
        AnalysisJob job = new AnalysisJob("job-101", Path.of("/evidence/suspect_01.mp4"));
        repository.save(job);

        Optional<AnalysisJob> found = repository.findById("job-101");
        assertTrue(found.isPresent());
        assertEquals("job-101", found.get().getId());
        assertEquals("suspect_01.mp4", found.get().getFileName());
    }

    @Test
    void testPersistsAcrossRepositoryInstances(@TempDir Path tempDir) {
        String database = tempDir.resolve("jobs.db").toString();
        SqliteJobRepository writer = new SqliteJobRepository(database);
        writer.save(new AnalysisJob("job-persisted", Path.of("video.mp4")));

        SqliteJobRepository reader = new SqliteJobRepository(database);
        assertEquals("video.mp4", reader.findById("job-persisted").orElseThrow().getFileName());
    }

    @Test
    @DisplayName("Should filter jobs by status")
    void testFindByStatus() {
        AnalysisJob job1 = new AnalysisJob("j1", Path.of("v1.mp4"));
        AnalysisJob job2 = new AnalysisJob("j2", Path.of("v2.mp4"));
        job2.updateProgress(ProcessingStage.EXTRACTING_FRAMES, 25.0);

        repository.save(job1);
        repository.save(job2);

        List<AnalysisJob> running = repository.findByStatus(JobStatus.RUNNING);
        assertEquals(1, running.size());
        assertEquals("j2", running.get(0).getId());

        List<AnalysisJob> pending = repository.findByStatus(JobStatus.PENDING);
        assertEquals(1, pending.size());
        assertEquals("j1", pending.get(0).getId());
    }

    @Test
    @DisplayName("Should search jobs by partial filename")
    void testSearchByFileName() {
        repository.save(new AnalysisJob("j1", Path.of("/data/deepfake_face_swap.mp4")));
        repository.save(new AnalysisJob("j2", Path.of("/data/authentic_interview.mp4")));

        List<AnalysisJob> results = repository.searchByFileName("deepfake");
        assertEquals(1, results.size());
        assertEquals("j1", results.get(0).getId());
    }

    @Test
    @DisplayName("Should delete job by ID")
    void testDeleteById() {
        AnalysisJob job = new AnalysisJob("j-del", Path.of("temp.mp4"));
        repository.save(job);
        assertEquals(1, repository.count());

        boolean deleted = repository.deleteById("j-del");
        assertTrue(deleted);
        assertEquals(0, repository.count());
        assertTrue(repository.findById("j-del").isEmpty());
    }
}
