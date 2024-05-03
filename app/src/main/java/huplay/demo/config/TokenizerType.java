package huplay.demo.config;

import huplay.demo.IdentifiedException;
import huplay.demo.tokenizer.*;

public enum TokenizerType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    SENTENCE_PIECE,
    SENTENCE_PIECE_BINARY;

    public static Tokenizer getTokenizer(Config config)
    {
        if (config.getTransformerType() == null)
        {
            throw new IdentifiedException("Tokenizer type isn't specified");
        }

        TokenizerType tokenizerType = TokenizerType.valueOf(config.getTokenizerType().toUpperCase());

        switch (tokenizerType)
        {
            case OPENAI_GPT_1: return new GPT1Tokenizer(config);
            case OPENAI_GPT_2: return new GPT2Tokenizer(config);
            case SENTENCE_PIECE: return new SentencePieceTokenizer(config);
            case SENTENCE_PIECE_BINARY: return new SentencePieceTokenizerBinaryConfig(config);
            default:
                throw new IdentifiedException("Unknown tokenizer type: " + config.getTokenizerType());
        }
    }
}
