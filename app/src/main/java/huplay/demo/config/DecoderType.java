package huplay.demo.config;

import huplay.demo.transformer.BaseDecoder;
import huplay.demo.transformer._2017_google_transformer.TransformerDecoder;
import huplay.demo.transformer._2021_eleuther_gptneo.NeoDecoder;
import huplay.demo.transformer._2022_huggingface_bloom.BloomDecoder;
import huplay.demo.transformer._2023_meta_llama.LlamaMHADecoder;
import huplay.demo.transformer._2023_meta_llama.LlamaGQADecoder;
import huplay.demo.transformer._2018_openai_gpt1.GPT1Decoder;
import huplay.demo.transformer._2019_openai_gpt2.GPT2Decoder;

public enum DecoderType
{
    ORIGINAL_DECODER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    HUGGING_FACE_BLOOM,
    ELEUTHERAI_NEO,
    META_LLAMA_MHA,
    META_LLAMA_GQA;

    public BaseDecoder getDecoder(int decoderId, Config config)
    {
        switch (this)
        {
            case ORIGINAL_DECODER: return new TransformerDecoder(config, decoderId);
            case OPENAI_GPT_1: return new GPT1Decoder(config, decoderId);
            case OPENAI_GPT_2: return new GPT2Decoder(config, decoderId);
            case HUGGING_FACE_BLOOM: return new BloomDecoder(config, decoderId);
            case ELEUTHERAI_NEO: return new NeoDecoder(config, decoderId);
            case META_LLAMA_MHA: return new LlamaMHADecoder(config, decoderId);
            case META_LLAMA_GQA: return new LlamaGQADecoder(config, decoderId);
            default:
                throw new RuntimeException("Unknown transformer type: " + config.getTransformerType());
        }
    }
}
