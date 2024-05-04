package huplay.demo.tokenizer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;
import huplay.demo.tokenizer.gpt.GPT1Tokenizer;
import huplay.demo.tokenizer.gpt.GPT2Tokenizer;
import huplay.demo.tokenizer.sentencePiece.SentencePieceTokenizer;

public enum TokenizerType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    SENTENCE_PIECE,
    TIKTOKEN;

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
            case TIKTOKEN:
            default:
                throw new IdentifiedException("Unknown tokenizer type: " + config.getTokenizerType());
        }
    }
}
