package huplay.demo.transformer;

import huplay.demo.config.ParameterReader;
import huplay.demo.config.ParameterType;
import huplay.demo.config.Config;

import java.util.*;

import static huplay.demo.App.UTIL;

public abstract class AbstractDecoder
{
    private final ParameterReader reader;
    private final String decoderParameterFormat;
    private final Map<String, String> parameterOverrides;

    protected final int decoderId;
    protected float attentionDividend;
    protected final int hiddenSize;
    protected final int headCount;
    protected final int headSize;
    protected final int feedForwardSize;
    protected final boolean lastDecoder;
    protected final float epsilon;

    private final Map<ParameterType, float[]> vectorParams = new HashMap<>();
    private final Map<ParameterType, float[][]> matrixParams = new HashMap<>();

    protected final List<float[][]> storedKeys = new ArrayList<>();
    protected final List<float[][]> storedValues = new ArrayList<>();

    public AbstractDecoder(Config config, int decoderId)
    {
        this.reader = config.getReader();
        this.decoderParameterFormat = config.getDecoderParameterFormat();
        this.parameterOverrides = config.getDecoderParameterOverrides();
        this.decoderId = decoderId;
        this.hiddenSize = config.getHiddenSize();
        this.headCount = config.getHeadCount();
        this.headSize = config.getHeadSize();
        this.feedForwardSize = config.getFeedForwardSize();
        this.lastDecoder = (decoderId == config.getDecoderCount());
        this.epsilon = config.getEpsilon();
    }

    /**
     * Process the input
     */
    public abstract float[] execute(float[] hiddenState, boolean isOutputProcessing);

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }

    protected void loadVector(ParameterType parameterType, String file, int size)
    {
        vectorParams.put(parameterType, reader.readVector(formatName(file), size));
    }

    protected void loadMatrix(ParameterType parameterType, String file, int rows, int cols)
    {
        matrixParams.put(parameterType, reader.readMatrix(formatName(file), rows, cols));
    }

    private float[] concat(float[] first, float[] second)
    {
        float[] both = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, both, first.length, second.length);
        return both;
    }
        
    protected float[] vector(ParameterType parameterType)
    {
        return vectorParams.get(parameterType);
    }

    protected float[][] matrix(ParameterType parameterType)
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

        String formattedName = decoderParameterFormat.replace("{decoderId}", "" + decoderId);
        return formattedName.replace("{name}", name);
    }
}
