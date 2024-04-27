package huplay.demo.transformer;

import huplay.demo.config.Config;

import java.util.*;

public abstract class BaseDecoder extends ParameterStore
{
    protected final int decoderId;
    protected float attentionDividend;
    protected final int hiddenSize;
    protected final int headCount;
    protected final int headSize;
    protected final int feedForwardSize;
    protected final boolean lastDecoder;
    protected final float epsilon;

    protected final List<float[][]> storedKeys = new ArrayList<>();
    protected final List<float[][]> storedValues = new ArrayList<>();

    public BaseDecoder(Config config, int decoderId)
    {
        super(config, config.getDecoderParameterFormat(), config.getDecoderParameterOverrides());
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

    @Override
    protected String formatName(String file)
    {
        if (parameterOverrides != null)
        {
            String override = parameterOverrides.get(file);
            if (override != null)
            {
                file = override;
            }
        }

        String formattedName = parameterFormat.replace("{decoderId}", "" + decoderId);
        return formattedName.replace("{name}", file);
    }
}
