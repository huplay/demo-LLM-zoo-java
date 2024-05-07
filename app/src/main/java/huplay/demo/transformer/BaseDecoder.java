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
        super(config);
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
    protected String formatName(String name)
    {
        if (config.getParameterNameOverrides() != null)
        {
            String override = config.getParameterNameOverrides().get(name);
            if (override != null)
            {
                name = override;
            }
        }

        String decoder = "" + decoderId;
        name = name.replace("{decoderId}", decoder);

        String formattedName = config.getDecoderNameFormat().replace("{decoderId}", decoder);
        formattedName = formattedName.replace("{name}", name);

        if (config.getParameterNameOverrides() != null)
        {
            String override = config.getParameterNameOverrides().get(formattedName);
            if (override != null)
            {
                formattedName = override;
            }
        }

        return formattedName;
    }
}
