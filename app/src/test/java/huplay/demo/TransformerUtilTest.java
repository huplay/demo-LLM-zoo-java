package huplay.demo;

import org.junit.Test;

import static huplay.demo.TransformerUtil.softmax;
import static org.junit.Assert.assertArrayEquals;

public class TransformerUtilTest
{
    @Test
    public void softmaxTest()
    {
        float[] values = new float[] {1, 2, 3, 4, 1, 2, 3};

        float[] expected = new float[] {0.023640543f, 0.06426166f, 0.1746813f, 0.474833f, 0.023640543f, 0.06426166f, 0.1746813f};

        assertArrayEquals(expected, softmax(values), 0);
    }
}
