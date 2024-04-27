package huplay.demo;

import huplay.demo.util.IndexedValue;

import java.util.List;

import static huplay.demo.AppLoader.UTIL;
import static java.lang.Math.*;

public class TransformerUtil
{
    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public static float gelu(float value)
    {
        // Using a constant for sqrt(2 / PI) didn't make it faster, most likely Java optimized it
        return (float) (0.5 * value * (1 + tanh(Math.sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }

    /**
     * SwiGLU activation function
     * Original paper: <a href="https://arxiv.org/abs/2002.05202" />
     */
    public static float swiglu(float value)
    {
        return (float) (value * (1f / (1f + Math.exp(-value))));
    }

    /**
     * Standard normalization with applying normalization weights and biases
     */
    public static float[] layerNorm(float[] vector, float[] weight, float[] bias, float epsilon)
    {
        // Standard normalization
        float[] result = UTIL.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < vector.length; i++)
        {
            result[i] = result[i] * weight[i] + (bias == null ? 0 : bias[i]);
        }

        return result;
    }

    /**
     * Root Mean Square Layer Normalization (RMS)
     * Original paper: <a href="https://arxiv.org/abs/1910.07467" />
     */
    public static float[] RMSLayerNorm(float[] vector, float[] weight, float epsilon)
    {
        int size = vector.length;

        // Calculate the sum of squares
        float sum = 0f;
        for (int i = 0; i < size; i++)
        {
            sum += vector[i] * vector[i];
        }

        // Calculate room mean square
        sum = 1f / sqrt(sum / vector.length + epsilon);

        //  Normalize and scale
        float[] result = new float[vector.length];

        for (int i = 0; i < size; i++)
        {
            result[i] = weight[i] * (sum * vector[i]);
        }

        return result;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(float[] vector)
    {
        float max = UTIL.max(vector);

        double total = 0;
        for (float value : vector)
        {
            double exp = exp(value - max);

            total = total + exp;
        }

        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            double exp = exp(vector[i] - max);

            ret[i] = (float) (exp / total);
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(List<IndexedValue> values)
    {
        float max = UTIL.max(values);

        double total = 0;
        for (IndexedValue value : values)
        {
            total = total + exp(value.getValue() - max);
        }

        float[] ret = new float[values.size()];

        for (int i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exp(values.get(i).getValue() - max) / total);
        }

        return ret;
    }

    public static float pow(float a, float b)
    {
        return (float)(Math.pow(a, b));
    }

    public static float sqrt(float value)
    {
        return (float)(Math.sqrt(value));
    }

    public static float cos(double value)
    {
        return (float)(Math.cos(value));
    }

    public static float sin(double value)
    {
        return (float)(Math.sin(value));
    }
}
