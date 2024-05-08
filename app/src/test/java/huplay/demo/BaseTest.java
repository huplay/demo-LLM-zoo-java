package huplay.demo;

import huplay.demo.util.FloatType;
import huplay.demo.util.Vector;

import static org.junit.Assert.*;

public class BaseTest
{
    protected Vector createVector(float... values)
    {
        Vector vector = new Vector(FloatType.FLOAT32, values.length);

        for (int i = 0; i < values.length; i ++)
        {
            vector.set(i, values[i]);
        }

        return vector;
    }

    protected void assertVectorEquals(float[] expected, Vector actual, float delta)
    {
        if (expected == null)
        {
            assertNull(actual);
        }
        else
        {
            assertNotNull(actual);

            assertEquals(expected.length, actual.size());

            for (int i = 0; i < expected.length; i++)
            {
                assertEquals(expected[i], actual.get(i), delta);
            }
        }
    }

    protected void assertMatrixEquals(float[][] expected, Vector[] actual, float delta)
    {
        if (expected == null)
        {
            assertNull(actual);
        }
        else
        {
            assertNotNull(actual);

            assertEquals(expected.length, actual.length);

            for (int i = 0; i < expected.length; i++)
            {
                assertVectorEquals(expected[i], actual[i], delta);
            }
        }
    }
}
