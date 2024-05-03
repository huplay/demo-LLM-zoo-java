package huplay.demo.transformer;

import huplay.demo.config.DecoderType;
import huplay.demo.config.Config;

import java.util.ArrayList;
import java.util.List;

public abstract class BaseTransformer extends ParameterStore
{
    protected final int decoderCount;
    protected final int hiddenSize;
    protected final int tokenCount;
    protected final int embeddingCount;
    protected final int contextSize;
    protected final float epsilon;

    protected final List<BaseDecoder> decoders = new ArrayList<>();

    public BaseTransformer(Config config, DecoderType decoderType)
    {
        super(config);
        this.decoderCount = config.getDecoderCount();
        this.hiddenSize = config.getHiddenSize();
        this.tokenCount = config.getTokenCount();
        this.embeddingCount = config.getTokenCount();
        this.contextSize = config.getContextSize();
        this.epsilon = config.getEpsilon();

        for (int i = 0; i < decoderCount; i++)
        {
            decoders.add(decoderType.getDecoder(i, config));
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
        for (BaseDecoder decoder : decoders)
        {
            decoder.clear();
        }
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

        if (config.getTransformerParameterNameFormat() != null)
        {
            name = config.getTransformerParameterNameFormat().replace("{name}", name);
        }

        return name;
    }

    @Override
    public long getParameterSize()
    {
        long parameterSize = super.getParameterSize();

        for (BaseDecoder decoder : decoders)
        {
            parameterSize += decoder.getParameterSize();
        }

        return parameterSize;
    }
}
