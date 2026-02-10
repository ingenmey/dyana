# base_analysis.py

from abc import ABC, abstractmethod
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches

class BaseAnalysis(ABC):
    def __init__(self, traj):
        self.traj = traj
        self.frame_idx = 0
        self.processed_frames = 0
        self.update_compounds = False
        self.allow_compound_update = True

    def skip_to_start(self):
        self.frame_idx = 0
        if self.start_frame > 1:
            print(f"Skipping forward to frame {self.start_frame}.")
            while self.frame_idx < self.start_frame - 1:
                self.traj.read_frame()
                self.frame_idx += 1

    def run(self):
        self.setup()
        self.setup_frame_loop()
        self.skip_to_start()

        while self.nframes != 0:
            try:
                if self.update_compounds:
                    self.traj.guess_molecules()
                    self.traj.update_molecule_coords()
                    if not self.post_compound_update():
                        self.frame_idx += 1
                        self.nframes -= 1
                        self.traj.read_frame()
                        continue
                else:
                    self.traj.update_molecule_coords()

                self.process_frame()
                self.processed_frames += 1

                if self.processed_frames % 100 == 0:
                    print(f"Processed {self.processed_frames} frames (current frame {self.frame_idx+1})")

                for _ in range(self.frame_stride):
                    self.frame_idx += 1
                    self.nframes -= 1
                    self.traj.read_frame()

            except ValueError:
                # End of trajectory file
                print("\nEnd of trajectory reached.")
                break

            except KeyboardInterrupt:
                # Graceful exit when user presses Ctrl+C
                print("\nInterrupt received! Exiting main loop and post-processing data...")
                break

        self.postprocess()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def post_compound_update(self):
        pass

    @abstractmethod
    def process_frame(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    def compound_selection(self, role="reference", multi=False, prompt_text=None):
        compounds = list(self.traj.compounds.values())
        print("\nAvailable Compounds:")
        for i, c in enumerate(compounds, start=1):
            print(f"{i}: {c.rep} (Number: {len(c.members)})")

        if multi:
            prompt_str = prompt_text or f"Choose the {role} compounds (comma-separated numbers): "
            choices = prompt(prompt_str).strip()
            idxs = [int(x.strip()) - 1 for x in choices.split(',') if x.strip()]
            return [(i, compounds[i]) for i in idxs]
        else:
            prompt_str = prompt_text or f"Choose the {role} compound (number): "
            idx = prompt_int(prompt_str, 1, minval=1) - 1
            return idx, compounds[idx]


    def atom_selection(self, role="reference", compound=None, prompt_text=None, allow_empty=False):
        """
        Select one or more atom labels from a compound.

        Parameters
        ----------
        role : str
            Role description for the prompt (default: "reference").
        compound : Compound, optional
            The compound to select atoms from.
        prompt_text : str, optional
            Custom prompt text. Can include {role}, {compound_num}, and {compound_name} placeholders.
        """
        if compound is not None:
            default_prompt = "Which atom(s) in {role} compound {compound_num} ({compound_name})? (comma-separated) "
        else:
            default_prompt = "Which atom(s) in {role} compound? (comma-separated) "

        prompt_str = (prompt_text or default_prompt).format(
            role=role,
            compound_num=compound.comp_id + 1 if compound else "",
            compound_name=compound.rep if compound else ""
        )

        if allow_empty:
            # Accept Enter as empty selection
            answer = prompt(prompt_str, default="")
        else:
            # Require at least one atom label
            answer = prompt(prompt_str)

        return [s.strip() for s in answer.split(',') if s.strip()]

    def setup_frame_loop(self):
        if self.allow_compound_update:
            self.update_compounds = prompt_yn("Perform molecule recognition and update compound list in each frame?", False)
        else:
            self.update_compounds = False
            print("\nPer-frame molecule recognition is disabled for this analysis.")
        self.start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
        self.nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
        self.frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)
