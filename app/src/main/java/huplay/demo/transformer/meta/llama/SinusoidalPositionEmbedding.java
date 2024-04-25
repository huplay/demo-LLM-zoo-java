package huplay.demo.transformer.meta.llama;

import huplay.demo.config.Config;

public class SinusoidalPositionEmbedding
{
    protected final float[][] transformMatrix;

    public SinusoidalPositionEmbedding(Config config)
    {
        this.transformMatrix = getTransformMatrix(config.getMaxLength(), config.getHiddenSize());
    }

    protected float[][] getTransformMatrix(int maxLength, int hiddenSize)
    {
        float[][] transformMatrix = new float[maxLength][hiddenSize];

        float[] positions = getSeries(maxLength);
        float[] progression = getProgression(maxLength);

        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int k = 0; k < hiddenSize / 2; k++)
            {
                int i = 2 * k;
                transformMatrix[pos][i] = (float) Math.sin(positions[i] * progression[k]);
                transformMatrix[pos][i + 1] = (float) Math.sin(positions[i + 1] * progression[k]);
            }
        }

        return transformMatrix;
    }

    public float[] toInput(float[] input, int pos)
    {
        float[] result = new float[input.length];

        for (int i = 0; i < input.length; i++)
        {
            result[i] = input[i] * transformMatrix[pos][i];
        }

        return result;
    }

    private float[] getSeries(int length)
    {
        float[] series = new float[length];
        for (int i = 0; i < length; i++) series[i] = i;
        return series;
    }

    private float[] getProgression(int length)
    {
        float[] progression = new float[length / 2];
        for (int i = 0; i < length / 2; i++) progression[i] = (float) Math.exp(-i * Math.log(10000) / length);
        return progression;
    }
}
