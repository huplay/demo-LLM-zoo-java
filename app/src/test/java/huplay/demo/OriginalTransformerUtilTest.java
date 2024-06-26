package huplay.demo;

import org.junit.Test;

import static huplay.demo.TransformerUtil.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class OriginalTransformerUtilTest
{
    @Test
    public void geluTest()
    {
        assertEquals(1.9545977115631104f, gelu(2f), 1e-15f);
        assertEquals(0.005039893556386232f, gelu(1e-2f), 1e-15f);
        assertEquals(-0.021692416f, gelu(-0.045f), 1e-15f);
    }

    @Test
    public void swigluTest()
    {
        assertEquals(1.7615942f, swiglu(2f), 1e-15f);
        assertEquals(0.005025f, swiglu(1e-2f), 1e-15f);
        assertEquals(-0.021993836f, swiglu(-0.045f), 1e-15f);
    }
/*
    @Test
    public void layerNormTest()
    {
        float[] weight = new float[]{0.5f, -1.234f, 1e-3f, 5};
        float[] bias = new float[]{0.012f, -4.234e-3f, 1e-3f, 2};
        float epsilon = 1e-5f;

        assertArrayEquals(new float[]{-0.6003678f, -0.004234f, 0.0022247357f},
                layerNorm(new float[]{1, 2, 3}, weight, bias, epsilon), 1e-15f);
    }

    @Test
    public void RSMLayerNormTest()
    {
        float[] weight = new float[]{0.5f, -1.234f, 1e-3f, 5};
        float epsilon = 1e-5f;

        assertArrayEquals(new float[]{0.23145477f, -1.1424607f, 0.0013887287f},
                RMSLayerNorm(new float[]{1, 2, 3}, weight, epsilon), 1e-15f);
    }

    @Test
    public void softmaxTest()
    {
        float[] values = new float[] {1, 2, 3, 4, 1, 2, 3};

        float[] expected = new float[] {0.023640543f, 0.06426166f, 0.1746813f, 0.474833f, 0.023640543f, 0.06426166f, 0.1746813f};

        assertArrayEquals(expected, softmax(values), 0);
    }*/
}
