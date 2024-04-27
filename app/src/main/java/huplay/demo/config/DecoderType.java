package huplay.demo.config;

import huplay.demo.transformer.BaseDecoder;
import huplay.demo.transformer.eleutherai.gptneo.NeoDecoder;
import huplay.demo.transformer.huggingface.bloom.BloomDecoder;
import huplay.demo.transformer.meta.llama.LlamaMHADecoder;
import huplay.demo.transformer.meta.llama.LlamaGQADecoder;
import huplay.demo.transformer.openai.gpt1.GPT1Decoder;
import huplay.demo.transformer.openai.gpt2.GPT2Decoder;

public enum DecoderType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    HUGGING_FACE_BLOOM,
    ELEUTHERAI_NEO,
    META_LLAMA_MHA,
    META_LLAMA_GQA;

    public BaseDecoder getDecoder(int decoderId, Config config, ParameterReader reader)
    {
        switch (this)
        {
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
