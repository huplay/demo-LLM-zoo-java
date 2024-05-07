package huplay.demo;

import java.io.PrintStream;

public class Logo
{
    private static final String[] LETTER_ROWS = new String[6];
    private static final int[] ANSI_COLOR_CODES;

    static
    {
        // The letters are based on the Standard font of this site: https://patorjk.com/software/taag
        // In this structure the "I" separates the letters, backslash is replaced by "x".
        LETTER_ROWS[0] = "I    I _ I _ _I   _  _   I  _  I _  __I  ___   I _ I  __I__  I      I       I   I       I   I    __I  ___  I _ I ____  I _____ I _  _   I ____  I  __   I _____ I  ___  I  ___  I   I   I  __I       I__  I ___ I   ____  I    _    I ____  I  ____ I ____  I _____ I _____ I  ____ I _   _ I ___ I     _ I _  __I _     I __  __ I _   _ I  ___  I ____  I  ___  I ____  I ____  I _____ I _   _ I__     __I__        __I__  __I__   __I _____I __ I__    I __ I /x I       I _ I       I _     I      I     _ I      I  __ I       I _     I _ I   _ I _    I _ I           I       I       I       I       I      I     I _   I       I       I          I      I       I     I   __I _ I__   I     I";
        LETTER_ROWS[1] = "I    I| |I( | I _| || |_ I | | I(_)/ /I ( _ )  I( )I / /Ix x I__/x__I   _   I   I       I   I   / /I / _ x I/ |I|___ x I|___ / I| || |  I| ___| I / /_  I|___  |I ( _ ) I / _ x I _ I _ I / /I _____ Ix x I|__ xI  / __ x I   / x   I| __ ) I / ___|I|  _ x I| ____|I|  ___|I / ___|I| | | |I|_ _|I    | |I| |/ /I| |    I|  x/  |I| x | |I / _ x I|  _ x I / _ x I|  _ x I/ ___| I|_   _|I| | | |Ix x   / /Ix x      / /Ix x/ /Ix x / /I|__  /I| _|Ix x   I|_ |I|/x|I       I( )I  __ _ I| |__  I  ___ I  __| |I  ___ I / _|I  __ _ I| |__  I(_)I  (_)I| | __I| |I _ __ ___  I _ __  I  ___  I _ __  I  __ _ I _ __ I ___ I| |_ I _   _ I__   __I__      __I__  __I _   _ I ____I  / /I| |Ix x  I     I";
        LETTER_ROWS[2] = "I    I| |I V VI|_  ..  _|I/ __)I  / / I / _ x/xI|/ I| | I | |Ix    /I _| |_ I   I _____ I   I  / / I| | | |I| |I  __) |I  |_ x I| || |_ I|___ x I| '_ x I   / / I / _ x I| (_) |I(_)I(_)I/ / I|_____|I x xI  / /I / / _` |I  / _ x  I|  _ x I| |    I| | | |I|  _|  I| |_   I| |  _ I| |_| |I | | I _  | |I| ' / I| |    I| |x/| |I|  x| |I| | | |I| |_) |I| | | |I| |_) |Ix___ x I  | |  I| | | |I x x / / I x x /x / / I x  / I x V / I  / / I| | I x x  I | |I    I       I x|I / _` |I| '_ x I / __|I / _` |I / _ xI| |_ I / _` |I| '_ x I| |I  | |I| |/ /I| |I| '_ ` _ x I| '_ x I / _ x I| '_ x I / _` |I| '__|I/ __|I| __|I| | | |Ix x / /Ix x /x / /Ix x/ /I| | | |I|_  /I | | I| |I | | I /x/|I";
        LETTER_ROWS[3] = "I    I|_|I    I|_      _|Ix__ xI / /_ I| (_>  <I   I| | I | |I/_  _xI|_   _|I _ I|_____|I _ I / /  I| |_| |I| |I / __/ I ___) |I|__   _|I ___) |I| (_) |I  / /  I| (_) |I x__, |I _ I _ Ix x I|_____|I / /I |_| I| | (_| |I / ___ x I| |_) |I| |___ I| |_| |I| |___ I|  _|  I| |_| |I|  _  |I | | I| |_| |I| . x I| |___ I| |  | |I| |x  |I| |_| |I|  __/ I| |_| |I|  _ < I ___) |I  | |  I| |_| |I  x V /  I  x V  V /  I /  x I  | |  I / /_ I| | I  x x I | |I    I       I   I| (_| |I| |_) |I| (__ I| (_| |I|  __/I|  _|I| (_| |I| | | |I| |I  | |I|   < I| |I| | | | | |I| | | |I| (_) |I| |_) |I| (_| |I| |   Ix__ xI| |_ I| |_| |I x V / I x V  V / I >  < I| |_| |I / /ˍI< <  I| |I  > >I|/x/ I";
        LETTER_ROWS[4] = "I    I(_)I    I  |_||_|  I(   /I/_/(_)I x___/x/I   I| | I | |I  x/  I  |_|  I( )I       I(_)I/_/   I x___/ I|_|I|_____|I|____/ I   |_|  I|____/ I x___/ I /_/   I x___/ I   /_/ I(_)I( )I x_xI       I/_/ I (_) I x x__,_|I/_/   x_xI|____/ I x____|I|____/ I|_____|I|_|    I x____|I|_| |_|I|___|I x___/ I|_|x_xI|_____|I|_|  |_|I|_| x_|I x___/ I|_|    I x__x_xI|_| x_xI|____/ I  |_|  I x___/ I   x_/   I   x_/x_/   I/_/x_xI  |_|  I/____|I| | I   x_xI | |I    I _____ I   I x____|I|____/ I x___|I x____|I x___|I|_|  I x__, |I|_| |_|I|_|I _/ |I|_|x_xI|_|I|_| |_| |_|I|_| |_|I x___/ I| .__/ I x__, |I|_|   I|___/I x__|I x____|I  x_/  I  x_/x_/  I/_/x_xI x__, |I/___|I | | I| |I | | I     I";
        LETTER_ROWS[5] = "I    I   I    I          I |_| I      I        I   I x_xI/_/ I      I       I|/ I       I   I      I       I   I       I       I        I       I       I       I       I       I   I|/ I    I       I    I     I  x____/ I         I       I       I       I       I       I       I       I     I       I      I       I        I       I       I       I       I       I       I       I       I         I            I      I       I      I|__|I      I|__|I    I|_____|I   I       I       I      I       I      I     I |___/ I       I   I|__/ I      I   I           I       I       I|_|    I    |_|I      I     I     I       I       I          I      I |___/ I     I  x_xI|_|I/_/  I     I";

        ANSI_COLOR_CODES = new int[] {30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97};
    }

    public static void showLogo(PrintStream OUT, String text, String textColors)
    {
        String[] logo = formatText(text, textColors);

        for (int i = 0; i < logo.length; i++)
        {
            if ((i == 0 || i < logo.length - 1) && hasNonSpace(logo[i])) // First and last line isn't displayed if empty
            {
                OUT.println(logo[i]);
            }
        }

        OUT.print("\033[0m");
    }

    private static boolean hasNonSpace(String line)
    {
        line = line.replace(" ", "");
        for (int i : ANSI_COLOR_CODES)
        {
            line = line.replace("\033[" + i + "m", "");
        }

        return line.length() > 0;
    }

    public static String[] formatText(String text, String textColors)
    {
        char[] characters = text.toCharArray();
        String[] colours = textColors.split(",");

        String[] result = new String[] {"", "", "", "", "", ""};

        for (int i = 0; i < characters.length; i++)
        {
            int colour = 0;
            if (colours.length > i && colours[i].length() > 0)
            {
                colour = Integer.parseInt(colours[i]);
            }

            result = appendLetter(result, getLetter(characters[i]), colour, true);
        }

        return result;
    }

    private static String[] getLetter(char letter)
    {
        String[] result = new String[6];

        for (int i = 0; i < result.length; i++)
        {
            // We have characters in the range of ASCII 32-126, which will be the index 0-93
            int letterIndex = letter - 32;
            if (letterIndex >= 0 && letterIndex <= 93)
            {
                int startIndex = indexOf(LETTER_ROWS[i], "I", letterIndex + 1);
                int endIndex = indexOf(LETTER_ROWS[i], "I", letterIndex + 2);
                result[i] = LETTER_ROWS[i].substring(startIndex + 1, endIndex).replace('x', '\\');
            }
        }

        return result;
    }

    private static String[] appendLetter(String[] text, String[] letter, int letterColor, boolean isKerning)
    {
        String[] result = new String[6];

        for (int i = 0; i < result.length; i++)
        {
            String color = "\033[" + ANSI_COLOR_CODES[letterColor] + "m";
            if (isKerning)
            {
                if (text[i].length() > 0)
                {
                    char last = text[i].charAt(text[i].length() - 1);
                    if (last == ' ')
                    {
                        result[i] = text[i].substring(0, text[i].length() - 1) + color + letter[i];
                    }
                    else
                    {
                        result[i] = text[i] + color + letter[i].substring(1);
                    }
                }
                else
                {
                    result[i] = color + letter[i];
                }
            }
            else
            {
                result[i] = text[i] + color + letter[i];
            }
        }

        return result;
    }

    private static int indexOf(String text, String searched, int nth)
    {
        int index = -1;
        while (nth > 0)
        {
            index = text.indexOf(searched, index + searched.length());
            if (index == -1)
            {
                return -1;
            }
            nth--;
        }

        return index;
    }
}
