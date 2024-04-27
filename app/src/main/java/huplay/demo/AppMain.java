package huplay.demo;

import huplay.demo.config.Arguments;
import huplay.demo.config.Config;
import huplay.demo.config.ParameterReader;
import huplay.demo.config.TransformerType;
import huplay.demo.tokenizer.Tokenizer;
import huplay.demo.transformer.BaseTransformer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static huplay.demo.AppLoader.*;

public class AppMain
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args) throws Exception
    {
        logo();

        Arguments arguments = readArguments(args);
        Config config = new Config(arguments);

        displayConfig(config, 0);

        OUT.print("\nLoading parameters... ");
        ParameterReader reader = new ParameterReader(config);
        config.setReader(reader);

        TransformerType transformerType = TransformerType.valueOf(config.getTransformerType());
        BaseTransformer transformer = transformerType.getTransformer(config);

        OUT.println("Done.");
        OUT.println("Parameter size:  " + Math.round((float) transformer.getParameterSize() / 1000_000) + "M");

        if (!config.isCalculationOnly())
        {
            Tokenizer tokenizer = Tokenizer.getInstance(config);
            Communicator processor = new Communicator(config, tokenizer, transformer);

            int pos = 0;
            int lastToken = config.getEndOfTextToken();

            while (true)
            {
                // Read the input text
                String inputText = input();

                List<Integer> inputTokens = new ArrayList<>();

                // If the input starts with "+" continue the same session
                if (inputText.equals("+")) inputTokens.add(lastToken);
                else if (inputText.startsWith("+"))
                {
                    inputTokens.addAll(tokenizer.encode(inputText.substring(1)));
                }
                else
                {
                    // Convert the input text into list of tokens
                    inputTokens = tokenizer.encode(inputText);

                    // Clear the transformer's stored values
                    pos = 0;
                    processor.clear();
                }

                // Use the Transformer
                List<Integer> outputTokens = processor.process(inputTokens, pos);

                // Convert the output to text and print it
                String response = tokenizer.decode(outputTokens);
                print(response, outputTokens, tokenizer);

                pos += outputTokens.size();
                lastToken = outputTokens.get(outputTokens.size() - 1);
            }
        }
    }

    public static void displayConfig(Config config, long parameterSize)
    {
        // Print settings
        OUT.println("Model: " + config.getName());
        OUT.println("Path: " + config.getModelPath());
        if (parameterSize > 0)
        {
            OUT.print("Number of parameters: " + Math.round(parameterSize / 1000_000d) + "M ");
        }
        OUT.println("Hidden size: " + config.getHiddenSize() +
                ", decoders: " + config.getDecoderCount() +
                ", heads: " + config.getHeadCount() +
                ", head size: " + config.getHeadSize());

        OUT.println("Maximum length of generated text: " + config.getLengthLimit());
        OUT.println("Output is selected from the best " + config.getTopK() + " tokens (topK)");
    }

    private static String input() throws IOException
    {
        OUT.print("\n\nInput text: ");
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        return reader.readLine();
    }

    private static void print(String response, List<Integer> outputTokens, Tokenizer tokenizer)
    {
        // The response was printed token by token, but for multi-token characters only "�" will be displayed

        // Here we recreate the token by token decoded response (which wasn't returned)
        StringBuilder tokenByTokenResponse = new StringBuilder();
        for (int token: outputTokens)
        {
            tokenByTokenResponse.append(tokenizer.decode(Collections.singletonList(token)));
        }

        // If the token by token decoded result is different to the final decoded result, print the corrected version
        if ( ! tokenByTokenResponse.toString().equals(response))
        {
            OUT.print("\nCorrected unicode response:\n" + response);
        }
    }
}
