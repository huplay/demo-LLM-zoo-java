package huplay.demo;

import huplay.demo.config.ModelConfig;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

public class Downloader implements Runnable
{
    private final ModelConfig modelConfig;
    private final String fileName;
    private final String path;

    private boolean isInProgress = true;
    private boolean isOk = true;

    public Downloader(ModelConfig modelConfig, String fileName, String path)
    {
        this.modelConfig = modelConfig;
        this.fileName = fileName;
        this.path = path;
    }

    @Override
    public void run()
    {
        try (FileOutputStream outputStream = new FileOutputStream(path + "/" + fileName))
        {
            URL url = new URL(determineDownloadUrl(modelConfig, fileName));

            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());
            FileChannel fileChannel = outputStream.getChannel();

            fileChannel.transferFrom(urlChannel, 0, Long.MAX_VALUE);
        }
        catch (IOException e)
        {
            isOk = false;
            throw new RuntimeException("IO error during file download: " + fileName + " error: " + e);
        }

        isInProgress = false;
    }

    private String determineDownloadUrl(ModelConfig modelConfig, String fileName)
    {
        String repoUrl = modelConfig.getRepo();

        if (repoUrl.startsWith("https://huggingface.co/"))
        {
            String branch = modelConfig.getBranch();

            if (branch == null || branch.equals(""))
            {
                branch = "main";
            }

            repoUrl += "/resolve/" + branch + "/" + fileName + "?download=true";
        }

        return repoUrl;
    }

    public boolean isInProgress()
    {
        return isInProgress;
    }

    public boolean isOk()
    {
        return isOk;
    }
}
