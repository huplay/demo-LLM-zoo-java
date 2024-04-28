package huplay.demo;

import huplay.demo.config.Config;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

public class Downloader implements Runnable
{
    private final Config config;
    private final String fileName;
    private final String path;

    private boolean isInProgress = true;
    private boolean isOk = true;

    public Downloader(Config config, String fileName, String path)
    {
        this.config = config;
        this.fileName = fileName;
        this.path = path;
    }

    @Override
    public void run()
    {
        try (FileOutputStream outputStream = new FileOutputStream(path))
        {
            URL url = new URL(determineDownloadUrl(config, fileName));

            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());
            FileChannel fileChannel = outputStream.getChannel();
            fileChannel.transferFrom(urlChannel, 0, Long.MAX_VALUE);
        }
        catch (IOException e)
        {
            isOk = false;
            throw new RuntimeException(e);
        }

        isInProgress = false;
    }

    private String determineDownloadUrl(Config config, String fileName)
    {
        String repoUrl = config.getParameterRepo();

        if (repoUrl.startsWith("https://huggingface.co/"))
        {
            String branch = config.getParameterRepoBranch();

            if (branch == null)
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
