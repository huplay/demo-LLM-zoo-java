package huplay.demo.tokenizer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Java implementation of Google SentencePiece tokenizer (without training)
 * Original implementation: <a href="https://github.com/google/sentencepiece" />
 *
 * This version uses the binary configuration (used by tinyLlamas)
 *
 * @author Hunor Szegi
 */
public class SentencePieceTokenizerBinaryConfig extends SentencePieceTokenizer
{
    public SentencePieceTokenizerBinaryConfig(Config config)
    {
        super(config);
    }

    @Override
    protected void init(Config config)
    {
        this.vocabulary = new String[config.getTokenCount()];
        this.vocabularyScores = new float[config.getTokenCount()];

        File tokenizerFile = config.getModelConfig().findFile("tokenizer.model");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile())
        {
            throw new IdentifiedException("SentencePiece tokenizer merges file is missing. (" + tokenizerFile.getName() + ")");
        }

        // Read the vocabulary from the binary config file
        Path configFilePath = Paths.get(tokenizerFile.getAbsolutePath());
        try (FileChannel channel = FileChannel.open(configFilePath, StandardOpenOption.READ))
        {
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // The first integer contains the maximum token length (we don't need this info)
            buffer.getInt();

            // Iterate over on all tokens
            for (int i = 0; i < config.getTokenCount(); i++)
            {
                // Read vocabulary score
                this.vocabularyScores[i] = buffer.getFloat();

                // Read token length
                int length = buffer.getInt();

                // Read token
                byte[] bytes = new byte[length];
                buffer.get(bytes);
                this.vocabulary[i] = new String(bytes, StandardCharsets.UTF_8);
            }
        }
        catch (Exception e)
        {
            throw new IdentifiedException("SentencePiece tokenizer vocabulary reading error.", e);
        }
    }
}