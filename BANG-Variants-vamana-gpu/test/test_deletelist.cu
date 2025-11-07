#include "../src/dynamic/deleteList.h"
#include <stdio.h>
#include <assert.h>

/**
 * Unit test for DeleteList class
 * Tests:
 * 1. Single point deletion
 * 2. Batch deletion
 * 3. Delete count tracking
 * 4. Clear operation
 */

void testSingleDeletion() {
    printf("\n=== Test 1: Single Point Deletion ===\n");

    DeleteList dl(10000);

    // Mark point 42 as deleted
    dl.markDeleted(42);

    // Verify it's deleted
    assert(dl.isDeleted(42) == true);
    printf("✓ Point 42 marked as deleted\n");

    // Verify delete count is 1
    assert(dl.getDeleteCount() == 1);
    printf("✓ Delete count is 1\n");

    // Verify another point is not deleted
    assert(dl.isDeleted(100) == false);
    printf("✓ Point 100 is not deleted\n");

    printf("Test 1 PASSED\n");
}

void testBatchDeletion() {
    printf("\n=== Test 2: Batch Deletion ===\n");

    DeleteList dl(10000);

    // Mark multiple points as deleted
    unsigned pointsToDelete[] = {10, 20, 30, 40, 50};
    unsigned count = 5;

    dl.batchMarkDeleted(pointsToDelete, count);

    // Verify all are deleted
    for (unsigned i = 0; i < count; i++) {
        assert(dl.isDeleted(pointsToDelete[i]) == true);
        printf("✓ Point %u marked as deleted\n", pointsToDelete[i]);
    }

    // Verify delete count is 5
    assert(dl.getDeleteCount() == 5);
    printf("✓ Delete count is 5\n");

    printf("Test 2 PASSED\n");
}

void testDuplicateDeletion() {
    printf("\n=== Test 3: Duplicate Deletion (Idempotent) ===\n");

    DeleteList dl(10000);

    // Mark point 42 as deleted twice
    dl.markDeleted(42);
    unsigned count1 = dl.getDeleteCount();
    printf("Delete count after first deletion: %u\n", count1);
    assert(count1 == 1);

    dl.markDeleted(42);
    unsigned count2 = dl.getDeleteCount();
    printf("Delete count after second deletion: %u\n", count2);

    // Count should still be 1 (idempotent)
    assert(count2 == 1);
    printf("✓ Duplicate deletion is idempotent\n");

    printf("Test 3 PASSED\n");
}

void testClearOperation() {
    printf("\n=== Test 4: Clear Operation ===\n");

    DeleteList dl(10000);

    // Mark some points as deleted
    unsigned pointsToDelete[] = {1, 2, 3, 4, 5};
    dl.batchMarkDeleted(pointsToDelete, 5);

    // Verify they're deleted
    assert(dl.getDeleteCount() == 5);
    assert(dl.isDeleted(3) == true);
    printf("✓ 5 points marked as deleted\n");

    // Clear all deletions
    dl.clear();

    // Verify all are now not deleted
    assert(dl.getDeleteCount() == 0);
    assert(dl.isDeleted(3) == false);
    printf("✓ All deletions cleared\n");

    printf("Test 4 PASSED\n");
}

void testLargeBatch() {
    printf("\n=== Test 5: Large Batch Deletion ===\n");

    unsigned N = 100000;
    DeleteList dl(N);

    // Create large batch of deletions
    unsigned batchSize = 1000;
    unsigned* pointsToDelete = new unsigned[batchSize];

    for (unsigned i = 0; i < batchSize; i++) {
        pointsToDelete[i] = i * 10; // Delete every 10th point
    }

    dl.batchMarkDeleted(pointsToDelete, batchSize);

    // Verify delete count
    assert(dl.getDeleteCount() == batchSize);
    printf("✓ Deleted %u points\n", batchSize);

    // Spot check some deletions
    assert(dl.isDeleted(0) == true);
    assert(dl.isDeleted(10) == true);
    assert(dl.isDeleted(100) == true);
    assert(dl.isDeleted(5) == false);
    assert(dl.isDeleted(15) == false);
    printf("✓ Spot checks passed\n");

    delete[] pointsToDelete;
    printf("Test 5 PASSED\n");
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║   DeleteList Unit Tests                   ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    testSingleDeletion();
    testBatchDeletion();
    testDuplicateDeletion();
    testClearOperation();
    testLargeBatch();

    printf("\n╔════════════════════════════════════════════╗\n");
    printf("║   ALL TESTS PASSED ✓                      ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    return 0;
}
