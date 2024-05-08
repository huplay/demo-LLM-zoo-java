package huplay.demo;

import huplay.demo.util.Util;
import huplay.demo.util.Vector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class UtilTest extends BaseTest
{
    private static final Util UTIL = new Util();

    @Test
    public void addVectorsTest()
    {
        Vector a = createVector(1, 2, 3, 4);
        Vector b = createVector(4, 5, 6, 7);
        float[] expectedResult = {5, 7, 9, 11};

        assertVectorEquals(expectedResult, UTIL.addVectors(a, b), 0);
    }

    @Test
    public void mulVectorByScalarTest()
    {
        Vector a = createVector(5, 6, 7, 8);
        float[] expectedResult = {15, 18, 21, 24};

        assertVectorEquals(expectedResult, UTIL.mulVectorByScalar(a, 3), 0);
    }

    @Test
    public void dotProductTest()
    {
        Vector a = createVector(5, 6, 7, 8);
        Vector b = createVector(4, 5, 6, 7);

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, UTIL.dotProduct(a, b), 0);
    }

    @Test
    public void mulVectorByMatrixTest()
    {
        Vector a = createVector(2, 5, 1, 8);

        Vector b1 = createVector(1, 0, 2, 0);
        Vector b2 = createVector(0, 3, 0, 4);
        Vector b3 = createVector(0, 0, 5, 0);
        Vector b4 = createVector(6, 0, 0, 7);
        Vector[] b = {b1, b2, b3, b4};

        float[] expectedResult = {4, 47, 5, 68};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void mulVectorByTransposedMatrixTest()
    {
        Vector a = createVector(5, 6, 7, 8);

        Vector b1 = createVector(1, 4, 7, 10);
        Vector b2 = createVector(2, 5, 8, 11);
        Vector b3 = createVector(3, 6, 9, 12);
        Vector[] b = {b1, b2, b3};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertVectorEquals(expectedResult, UTIL.mulVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void splitVectorTest()
    {
        Vector vector = createVector(1, 2, 3, 4, 5, 6);
        float[][] expectedResult = {{1, 2}, {3, 4}, {5, 6}};

        assertMatrixEquals(expectedResult, UTIL.splitVector(vector, 3), 0);
    }

    @Test
    public void flattenMatrixTest()
    {
        Vector v1 = createVector(1, 2);
        Vector v2 = createVector(3, 4);
        Vector v3 = createVector(5, 6);
        Vector[] matrix = {v1, v2, v3};

        float[] expectedResult = {1, 2, 3, 4, 5, 6};

        assertVectorEquals(expectedResult, UTIL.flattenMatrix(matrix), 0);
    }

    @Test
    public void averageTest()
    {
        Vector vector = createVector(1, 2, 3, 4, 5, 6);

        assertEquals(3.5f, UTIL.average(vector), 0);
    }
}
