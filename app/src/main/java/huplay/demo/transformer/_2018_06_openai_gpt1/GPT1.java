package huplay.demo.transformer._2018_06_openai_gpt1;

import huplay.demo.transformer.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.config.ParameterType.*;

/**
  OpenAI GPT-1 transformer

  Difference to the original transformer:
    - Learned position embedding (not sinusoid)

 * @author Hunor Szegi
 */
public class GPT1 extends BaseTransformer
{
    public GPT1(Config config)
    {
        super(config, DecoderType.OPENAI_GPT_1);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "tokens_embed.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "positions_embed.weight", contextSize, hiddenSize);
    }

    public Vector execute(int pos, int token, boolean isOutputProcessing)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS)[token];

        // Position embedding
        hiddenState = UTIL.addVectors(hiddenState, matrix(POSITION_EMBEDDINGS)[pos]);

        // Decoder stack
        for (BaseDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        return hiddenState;
    }
}
