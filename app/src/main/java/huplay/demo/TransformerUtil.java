package huplay.demo;

import huplay.demo.util.IndexedValue;
import huplay.demo.util.Vector;

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
    public static Vector layerNorm(Vector vector, Vector weight, Vector bias, float epsilon)
    {
        // Standard normalization
        Vector result = UTIL.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < vector.size(); i++)
        {
            result.set(i, result.get(i) * weight.get(i) + bias.get(i));
        }

        return result;
    }

    /**
     * Root Mean Square Layer Normalization (RMS)
     * Original paper: <a href="https://arxiv.org/abs/1910.07467" />
     */
    public static Vector RMSLayerNorm(Vector vector, Vector weight, float epsilon)
    {
        int size = vector.size();

        // Calculate the sum of squares
        float sum = 0f;
        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            sum += value * value;
        }

        // Calculate room mean square
        sum = 1f / sqrt(sum / size + epsilon);

        //  Normalize and scale
        Vector result = new Vector(vector.getFloatType(), size);

        for (int i = 0; i < size; i++)
        {
            result.set(i, weight.get(i) * (sum * vector.get(i)));
        }

        return result;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static Vector softmax(Vector vector)
    {
        float max = UTIL.max(vector);

        double total = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            double exp = exp(value - max);

            total = total + exp;
        }

        Vector ret = new Vector(vector.getFloatType(), vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            double exp = exp(vector.get(i) - max);

            ret.set(i, (float) (exp / total));
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
