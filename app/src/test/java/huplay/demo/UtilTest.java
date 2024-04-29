package huplay.demo;

import huplay.demo.util.Util;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class UtilTest
{
    private static final Util UTIL = new Util();

    @Test
    public void addVectorsTest()
    {
        float[] a = {1, 2, 3, 4};
        float[] b = {4, 5, 6, 7};
        float[] expectedResult = {5, 7, 9, 11};

        assertArrayEquals(expectedResult, UTIL.addVectors(a, b), 0);
    }

    @Test
    public void mulVectorByScalarTest()
    {
        float[] a = {5, 6, 7, 8};
        float[] expectedResult = {15, 18, 21, 24};

        assertArrayEquals(expectedResult, UTIL.mulVectorByScalar(a, 3), 0);
    }

    @Test
    public void dotProductTest()
    {
        float[] a = {5, 6, 7, 8};
        float[] b = {4, 5, 6, 7};

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, UTIL.dotProduct(a, b), 0);
    }

    @Test
    public void mulVectorByMatrixTest()
    {
        float[] a = {2, 5, 1, 8};
        float[][] b = {
                {1, 0, 2, 0},
                {0, 3, 0, 4},
                {0, 0, 5, 0},
                {6, 0, 0, 7}};

        float[] expectedResult = {4, 47, 5, 68};

        assertArrayEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void mulVectorByTransposedMatrixTest()
    {
        float[] a = {5, 6, 7, 8};
        float[][] b = {
                {1, 4, 7, 10},
                {2, 5, 8, 11},
                {3, 6, 9, 12}};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertArrayEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void splitVectorTest()
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};
        float[][] expectedResult = {{1, 2}, {3, 4}, {5, 6}};

        assertArrayEquals(expectedResult, UTIL.splitVector(matrix, 3));
    }

    @Test
    public void flattenMatrixTest()
    {
        float[][] matrix = {{1, 2}, {3, 4}, {5, 6}};
        float[] expectedResult = {1, 2, 3, 4, 5, 6};

        assertArrayEquals(expectedResult, UTIL.flattenMatrix(matrix), 0);
    }

    @Test
    public void averageTest()
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};

        assertEquals(3.5f, UTIL.average(matrix), 0);
    }
}
