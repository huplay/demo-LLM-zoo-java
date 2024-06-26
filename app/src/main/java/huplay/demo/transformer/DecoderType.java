package huplay.demo.transformer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;
import huplay.demo.transformer._2018_01_google_transformer.OriginalTransformerDecoder;
import huplay.demo.transformer._2021_03_eleuther_gptneo.GPTNeoDecoder;
import huplay.demo.transformer._2021_06_eleuther_gptj.GPTJDecoder;
import huplay.demo.transformer._2022_05_big_science_bloom.BloomDecoder;
import huplay.demo.transformer._2023_02_meta_llama.LlamaDecoder;
import huplay.demo.transformer._2018_06_openai_gpt1.GPT1Decoder;
import huplay.demo.transformer._2019_02_openai_gpt2.GPT2Decoder;

public enum DecoderType
{
    ORIGINAL_DECODER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    BIG_SCIENCE_BLOOM,
    ELEUTHERAI_GPT_NEO,
    ELEUTHERAI_GPT_J,
    META_LLAMA;

    public BaseDecoder getDecoder(int decoderId, Config config)
    {
        switch (this)
        {
            case ORIGINAL_DECODER: return new OriginalTransformerDecoder(config, decoderId);
            case OPENAI_GPT_1: return new GPT1Decoder(config, decoderId);
            case OPENAI_GPT_2: return new GPT2Decoder(config, decoderId);
            case BIG_SCIENCE_BLOOM: return new BloomDecoder(config, decoderId);
            case ELEUTHERAI_GPT_NEO: return new GPTNeoDecoder(config, decoderId);
            case ELEUTHERAI_GPT_J: return new GPTJDecoder(config, decoderId);
            case META_LLAMA: return new LlamaDecoder(config, decoderId);
            default:
                throw new IdentifiedException("No decoder implementation for this type: " + this.name());
        }
    }
}
