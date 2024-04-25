package huplay.demo.transformer;

import huplay.demo.config.DecoderType;
import huplay.demo.config.ParameterType;
import huplay.demo.config.ParameterReader;
import huplay.demo.config.Config;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class AbstractTransformer
{
    private final ParameterReader reader;
    private final String transformerParameterFormat;
    private final Map<String, String> parameterOverrides;

    protected final int decoderCount;
    protected final int hiddenSize;
    protected final int tokenCount;
    protected final int maxLength;
    protected final float epsilon;

    protected final List<AbstractDecoder> decoders = new ArrayList<>();

    private final Map<ParameterType, float[]> vectorParams = new HashMap<>();
    private final Map<ParameterType, float[][]> matrixParams = new HashMap<>();

    public AbstractTransformer(Config config, DecoderType decoderType)
    {
        this.reader = config.getReader();
        this.transformerParameterFormat = config.getTransformerParameterFormat();
        this.parameterOverrides = config.getTransformerParameterOverrides();
        this.decoderCount = config.getDecoderCount();
        this.hiddenSize = config.getHiddenSize();
        this.tokenCount = config.getTokenCount();
        this.maxLength = config.getMaxLength();
        this.epsilon = config.getEpsilon();

        for (int i = 0; i < decoderCount; i++)
        {
            decoders.add(decoderType.getDecoder(i, config, reader));
        }
    }

    /**
     * Process a single token
     */
    public abstract float[] execute(int pos, float[] embedding, boolean isOutputProcessing);

    /**
     * Clear stored values in all decoders to start a new session
     */
    public void clear()
    {
        for (AbstractDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }

    protected void loadVector(ParameterType parameterType, String file, int size)
    {
        vectorParams.put(parameterType, reader.readVector(formatName(file), size));
    }

    protected void loadMatrix(ParameterType parameterType, String file, int rows, int cols)
    {
        matrixParams.put(parameterType, reader.readMatrix(formatName(file), rows, cols));
    }

    public float[] vector(ParameterType parameterType)
    {
        return vectorParams.get(parameterType);
    }

    public float[][] matrix(ParameterType parameterType)
    {
        return matrixParams.get(parameterType);
    }

    private String formatName(String name)
    {
        if (parameterOverrides != null)
        {
            String override = parameterOverrides.get(name);
            if (override != null)
            {
                name = override;
            }
        }

        return transformerParameterFormat.replace("{name}", name);
    }
}
