package huplay.demo.tokenizer.gpt;

import huplay.demo.IdentifiedException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class MergesReader
{
    public static Map<Pair, Integer> readMergesFile(File file, boolean isOmitFirstLine)
    {
        Map<Pair, Integer> merges = new HashMap<>(50000);

        try
        {
            FileInputStream inputStream = new FileInputStream(file);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            if (isOmitFirstLine) reader.readLine();

            int i = 0;
            while (true)
            {
                String line = reader.readLine();

                if (line == null) break;

                String[] pairs = line.split(" ");
                merges.put(new Pair(pairs[0], pairs[1]), i);

                i++;
            }

            reader.close();
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot read merges file: " + file.getName(), e);
        }

        return merges;
    }
}
