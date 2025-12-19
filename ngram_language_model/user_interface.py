from ngram import NgramCharacterModel
import curses
import re
import sys
import os
from typing import List


class TerminalUI:
    def __init__(self, prediction_model, text_content=None):
        self.screen = None
        self.suggestions = []
        self.current_suggestion_idx = 0
        self.scores = []
        self.text_content = text_content
        self.user_input = ""
        self.cursor_pos = 0
        self.cursor_row = 1
        self.cursor_col = 0

        self.suggestions_panel = None
        self.text_panel = None
        self.input_panel = None
        self.scores_panel = None

        self.prediction_model = prediction_model

        # Track typed letters per current word
        self.current_word_keystrokes = 0

        # List of (k_i, l_i) for each completed word:
        #   k_i = letters typed for that word
        #   l_i = number of alphabetic letters in the finalized word
        self.word_stats = []

        # Tab key counter
        self.tabKeyCount = 0
    def finalize_current_word_stats(self) -> None:
        """
        Called whenever a word is 'finished' (typing space or pressing Enter).
        Records how many letters were typed (k_i) vs. how many letters
        the final word has (l_i).
        """
        words = self.user_input.strip().split()
        if not words:
            # No word typed yet, reset
            self.current_word_keystrokes = 0
            return

        last_word = words[-1]
        # Count how many alphabetic letters are in the final word
        l_i = sum(1 for c in last_word if c.isalpha())
        k_i = self.current_word_keystrokes

        if l_i > 0:
            self.word_stats.append((k_i, l_i))

        # Reset for next word
        self.current_word_keystrokes = 0


    def calculate_scores(self, text: str) -> List[int]:
        # 1) Total letters typed across all completed words
        total_typed_letters = sum(k for (k, _) in self.word_stats)

        # 2) Total tab presses
        total_tab_keys = self.tabKeyCount

        # sum(l_i) across all completed words
        sum_of_final_letters = sum(l for (_, l) in self.word_stats)

        # 3) avg_letters_per_word
        if sum_of_final_letters > 0:
            avg_letters_per_word = total_typed_letters / sum_of_final_letters
        else:
            avg_letters_per_word = 0.0

        # How many words have been completed so far
        total_words = len(self.word_stats)

        # 4) avg_tabs_per_word
        if total_words > 0:
            avg_tabs_per_word = total_tab_keys / total_words
        else:
            avg_tabs_per_word = 0.0

        return [
            total_typed_letters,    # 1
            total_tab_keys,         # 2
            avg_letters_per_word,   # 3
            avg_tabs_per_word       # 4
        ]

    def find_last_word_start(self, text: str, cursor_pos: int) -> int:
        """Find the start position of the last word being typed."""
        if cursor_pos == 0:
            return 0

        text_before_cursor = text[:cursor_pos]
        match = re.search(r"[^\s]*$", text_before_cursor)
        if match:
            return cursor_pos - len(match.group(0))
        return cursor_pos

    def get_current_word(self) -> str:
        """Get the current word being typed."""
        word_start = self.find_last_word_start(self.user_input, self.cursor_pos)
        return self.user_input[word_start : self.cursor_pos]

    def replace_current_word(self, new_word: str) -> None:
        """Replace the current word with a suggestion."""
        word_start = self.find_last_word_start(self.user_input, self.cursor_pos)
        self.user_input = (
            self.user_input[:word_start] + new_word + self.user_input[self.cursor_pos :]
        )
        self.cursor_pos = word_start + len(new_word)
    

    def draw_suggestions_panel(self) -> None:
        """Draw the suggestions panel (top panel)."""
        h, w = self.suggestions_panel.getmaxyx()
        self.suggestions_panel.erase()

        self.suggestions_panel.box()
        self.suggestions_panel.addstr(0, 2, " Suggestions ")

        if not self.suggestions:
            self.suggestions_panel.addstr(1, 2, "No suggestions")
        else:
            display_text = ""
            for i, suggestion in enumerate(self.suggestions):
                if i == self.current_suggestion_idx:
                    display_text += f"[{suggestion}] "
                else:
                    display_text += f"{suggestion} "

            if len(display_text) > w - 4:
                display_text = display_text[: w - 7] + "..."

            self.suggestions_panel.addstr(1, 2, display_text)

        self.suggestions_panel.noutrefresh()

    def draw_text_panel(self) -> None:
        """Draw the text panel (second panel)."""
        h, w = self.text_panel.getmaxyx()
        self.text_panel.erase()

        self.text_panel.box()
        self.text_panel.addstr(0, 2, " Text Content ")

        words = self.text_content.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) > w - 4:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word

        if current_line:
            lines.append(current_line)

        for i, line in enumerate(lines):
            if i < h - 2:
                self.text_panel.addstr(i + 1, 2, line)

        self.text_panel.noutrefresh()

    def draw_input_panel(self) -> None:
        """Draw the input panel (third panel)."""
        h, w = self.input_panel.getmaxyx()
        self.input_panel.erase()

        self.input_panel.box()
        self.input_panel.addstr(0, 2, " Input ")

        prompt = "> "
        prompt_len = len(prompt)

        available_width = w - 4
        first_line_width = available_width - prompt_len
        
        lines = []
        current_pos = 0
        text = self.user_input
        
        first_line_text = text[:first_line_width] if len(text) > 0 else ""
        lines.append(first_line_text)
        current_pos = len(first_line_text)
        
        while current_pos < len(text) and len(lines) < h - 2:
            next_chunk = text[current_pos:current_pos + available_width]
            lines.append(next_chunk)
            current_pos += len(next_chunk)
        
        for i, line in enumerate(lines):
            if i >= h - 2:
                break
            if i == 0:
                self.input_panel.addstr(i + 1, 2, prompt + line)
            else:
                self.input_panel.addstr(i + 1, 2, line)
        
        cursor_pos = self.cursor_pos
        if current_pos <= first_line_width:
            cursor_y = 1
            cursor_x = 2 + prompt_len + cursor_pos
        else:
            cursor_pos -= first_line_width
            cursor_y = 1 + (cursor_pos // available_width) + 1
            cursor_x = 2 + (cursor_pos % available_width)
            
        cursor_y = min(cursor_y, h - 2)
        cursor_x = min(cursor_x, w - 2)
        
        self.cursor_row = cursor_y
        self.cursor_col = cursor_x
        
        try:
            self.input_panel.move(cursor_y, cursor_x)
        except:
            self.input_panel.move(1, 2 + prompt_len)

        self.input_panel.noutrefresh()

    def draw_scores_panel(self) -> None:
        """Draw the scores panel (bottom panel)."""
        h, w = self.scores_panel.getmaxyx()
        self.scores_panel.erase()

        self.scores_panel.box()
        self.scores_panel.addstr(0, 2, " Scores ")

        # TODO: Set score labels
        self.scores = self.calculate_scores(self.user_input)

        score_labels = [
            "Letter Keys",        # total letters typed
            "Tab Keys",           # total tab presses
            "Avg Letters/Word",   # sum(k_i)/sum(l_i)
            "Avg Tabs/Word",      # total_tab_keys / total_words
        ]

        display_text = " | ".join(
            f"{label} {score}" for label, score in zip(score_labels, self.scores)
        )

        if len(display_text) > w - 4:
            display_text = display_text[: w - 7] + "..."

        self.scores_panel.addstr(1, (w - len(display_text)) // 2, display_text)
        self.scores_panel.noutrefresh()

    def handle_input(self, key) -> bool:
        if key == curses.KEY_RESIZE:
            return True
            
        if key == 27:  # ESC key
            return False  # End the loop
            
        if key == 9:  # Tab key
            self.tabKeyCount += 1
            if self.suggestions:
                self.current_suggestion_idx = (self.current_suggestion_idx + 1) % len(self.suggestions)
            return True
            
        if key == 10:  # Enter key -> accept the current suggestion if available
            if self.suggestions and self.current_suggestion_idx < len(self.suggestions):
                self.replace_current_word(self.suggestions[self.current_suggestion_idx])
            # Finalize the word because user accepted a suggestion or pressed Enter
            self.finalize_current_word_stats()
            self.suggestions = []
            self.current_suggestion_idx = 0
            return True

        # Backspace - handling multiple key codes for Windows compatibility
        if key in (curses.KEY_BACKSPACE, 127, 8):
            if self.cursor_pos > 0:
                char_deleted = self.user_input[self.cursor_pos - 1]
                self.user_input = (
                    self.user_input[:self.cursor_pos - 1] + self.user_input[self.cursor_pos:]
                )
                self.cursor_pos -= 1
                # If deleted char is alpha, decrement our typed letter count for the current word
                if char_deleted.isalpha() and self.current_word_keystrokes > 0:
                    self.current_word_keystrokes -= 1

                current_word = self.get_current_word()
                self.suggestions = self.prediction_model.predict_top_words(current_word)
                self.current_suggestion_idx = 0
            return True

        # Move cursor left
        if key == curses.KEY_LEFT:
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
                current_word = self.get_current_word()
                self.suggestions = self.prediction_model.predict_top_words(current_word)
                self.current_suggestion_idx = 0
            return True

        # Move cursor right
        if key == curses.KEY_RIGHT:
            if self.cursor_pos < len(self.user_input):
                self.cursor_pos += 1
                current_word = self.get_current_word()
                self.suggestions = self.prediction_model.predict_top_words(current_word)
                self.current_suggestion_idx = 0
            return True

        # Normal printable character
        if 32 <= key <= 126:
            char = chr(key)
            self.user_input = (
                self.user_input[:self.cursor_pos] + char + self.user_input[self.cursor_pos:]
            )
            self.cursor_pos += 1

            # If it's alphabetical, count it as a typed letter
            if char.isalpha():
                self.current_word_keystrokes += 1

            # If it's space, the user finished a word
            if char == ' ':
                self.finalize_current_word_stats()

            # Update suggestions for the new current word
            current_word = self.get_current_word()
            self.suggestions = self.prediction_model.predict_top_words(current_word)
            self.current_suggestion_idx = 0

        return True

    def run(self) -> None:
        """Main function to run the terminal UI."""
        try:
            self.screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.start_color()
            self.screen.keypad(True)

            curses.curs_set(1)

            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)

            max_y, max_x = self.screen.getmaxyx()

            suggestions_height = 3
            text_height = (max_y - 6) // 2
            input_height = (max_y - 6) // 2
            scores_height = 3

            self.suggestions_panel = curses.newwin(suggestions_height, max_x, 0, 0)
            self.text_panel = curses.newwin(text_height, max_x, suggestions_height, 0)
            self.input_panel = curses.newwin(
                input_height, max_x, suggestions_height + text_height, 0
            )
            self.scores_panel = curses.newwin(
                scores_height, max_x, suggestions_height + text_height + input_height, 0
            )

            self.draw_suggestions_panel()
            self.draw_text_panel()
            self.draw_input_panel()
            self.draw_scores_panel()

            self.input_panel.move(1, 4)
            curses.doupdate()

            running = True
            while running:
                try:
                    self.input_panel.move(self.cursor_row, self.cursor_col)
                except:
                    self.input_panel.move(1, 4)
                self.input_panel.noutrefresh()
                curses.doupdate()

                key = self.screen.getch()
                running = self.handle_input(key)

                if key == curses.KEY_RESIZE:
                    max_y, max_x = self.screen.getmaxyx()

                    suggestions_height = 3
                    text_height = (max_y - 6) // 2 + 2
                    input_height = (max_y - 6) // 2 + 1
                    scores_height = 3

                    self.suggestions_panel = curses.newwin(
                        suggestions_height, max_x, 0, 0
                    )
                    self.text_panel = curses.newwin(
                        text_height, max_x, suggestions_height, 0
                    )
                    self.input_panel = curses.newwin(
                        input_height, max_x, suggestions_height + text_height, 0
                    )
                    self.scores_panel = curses.newwin(
                        scores_height,
                        max_x,
                        suggestions_height + text_height + input_height,
                        0,
                    )

                self.draw_suggestions_panel()
                self.draw_text_panel()
                self.draw_input_panel()
                self.draw_scores_panel()

                try:
                    self.input_panel.move(self.cursor_row, self.cursor_col)
                    self.input_panel.noutrefresh()
                except:
                    pass
                curses.doupdate()

        finally:
            if self.screen:
                curses.nocbreak()
                self.screen.keypad(False)
                curses.echo()
                curses.endwin()
















if __name__  == "__main__":
    # Usage: python user_interface.py <path_to_training_corpus> [--auto]
    if len(sys.argv) < 2:
        print("Usage: python user_interface.py <path_to_training_corpus> [--auto]")
        sys.exit(1)

    trainingCorpusPath = sys.argv[1]
    auto_mode = False
    if len(sys.argv) > 2 and sys.argv[2] == "--auto":
        auto_mode = True

    trainingCorpus = ""
    if os.path.isdir(trainingCorpusPath):
        corpusFilePathList = sorted(os.listdir(trainingCorpusPath))
        for filename in corpusFilePathList:
            fullPath = os.path.join(trainingCorpusPath, filename)
            try:
                # Added explicit encoding for Windows compatibility
                with open(fullPath, "r", encoding="utf-8") as file:
                    trainingCorpus += file.read()
            except FileNotFoundError:
                print(f"File not found: {fullPath}")
                sys.exit(1)
    else:
        try:
            # Added explicit encoding for Windows compatibility
            with open(trainingCorpusPath, "r", encoding="utf-8") as file:
                trainingCorpus = file.read()
        except FileNotFoundError:
            print(f"File not found: {trainingCorpusPath}")
            sys.exit(1)

    try:
        # Added explicit encoding for Windows compatibility
        with open("text_content.txt", "r", encoding="utf-8") as file:
            textContent = file.read()
    except FileNotFoundError:
        print("File 'text_content.txt' not found!")
        sys.exit(1)

    # You can set n to whatever you like (2,3,...,10)
    n = 2
    model = NgramCharacterModel(trainingCorpus, n)
    ui = TerminalUI(model, text_content=textContent)
    ui.run()
    
