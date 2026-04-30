# base_analysis.py

from abc import ABC, abstractmethod

from config_builder import prompt_config_from_schema
from config_schema import FrameLoopConfig
from input_providers import InteractiveInputProvider

class BaseAnalysis(ABC):
    CONFIG_CLASS = None
    CONFIG_SCHEMA = None

    def __init__(self, traj, input_provider=None):
        self.traj = traj
        self.input_provider = input_provider or InteractiveInputProvider()
        self.frame_idx = 0
        self.processed_frames = 0
        self.update_compounds = False
        self.allow_compound_update = True
        self.start_frame = 1
        self.nframes = -1
        self.frame_stride = 1
        self._analysis_configured = False
        self._frame_loop_configured = False

    def get_input_provider(self, provider=None):
        return provider or self.input_provider

    def get_compounds(self):
        return list(self.traj.compounds.values())

    def compound_by_index(self, index):
        compounds = self.get_compounds()
        return compounds[index]

    def skip_to_start(self):
        self.frame_idx = 0
        if self.start_frame > 1:
            print(f"Skipping forward to frame {self.start_frame}.")
            while self.frame_idx < self.start_frame - 1:
                self.traj.read_frame()
                self.frame_idx += 1

    def _execute_frame_loop(self):
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

    def run(self):
        if not self._analysis_configured:
            self.setup()
        if not self._frame_loop_configured:
            self.configure_frame_loop(self.prompt_frame_loop_config())
        self.skip_to_start()
        self._execute_frame_loop()
        self.postprocess()

    def setup(self):
        if self.CONFIG_CLASS is None:
            raise NotImplementedError(f"{type(self).__name__} must override setup() or define CONFIG_CLASS.")
        self.configure(self.prompt_config())
        self._analysis_configured = True

    def prompt_config(self, provider=None):
        if self.CONFIG_CLASS is None or self.CONFIG_SCHEMA is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override prompt_config() or define CONFIG_CLASS and CONFIG_SCHEMA."
            )

        return prompt_config_from_schema(self, self.CONFIG_SCHEMA, self.CONFIG_CLASS, provider=provider)

    def configure_frame_loop(self, frame_loop):
        self.update_compounds = bool(frame_loop.update_compounds) if self.allow_compound_update else False
        self.start_frame = frame_loop.start_frame
        self.nframes = frame_loop.nframes
        self.frame_stride = frame_loop.frame_stride
        self._frame_loop_configured = True
        return self

    def mark_configured(self):
        self._analysis_configured = True
        return self

    def prompt_frame_loop_config(self, provider=None):
        input_provider = self.get_input_provider(provider)
        if self.allow_compound_update:
            update_compounds = input_provider.ask_bool(
                "Perform molecule recognition and update compound list in each frame?",
                False,
            )
        else:
            update_compounds = False
            print("\nPer-frame molecule recognition is disabled for this analysis.")

        return FrameLoopConfig(
            update_compounds=update_compounds,
            start_frame=input_provider.ask_int(
                "In which trajectory frame to start processing the trajectory?",
                1,
                minval=1,
            ),
            nframes=input_provider.ask_int(
                "How many trajectory frames to read (from this position on)?",
                -1,
                "all",
            ),
            frame_stride=input_provider.ask_int(
                "Use every n-th read trajectory frame for the analysis:",
                1,
                minval=1,
            ),
        )

    @abstractmethod
    def configure(self, config):
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

    def compound_selection(self, role="reference", multi=False, prompt_text=None, provider=None):
        input_provider = self.get_input_provider(provider)
        compounds = self.get_compounds()
        print("\nAvailable Compounds:")
        for i, c in enumerate(compounds, start=1):
            print(f"{i}: {c.rep} (Number: {len(c.members)})")

        if multi:
            prompt_str = prompt_text or f"Choose the {role} compounds (comma-separated numbers): "
            choices = input_provider.ask_str(prompt_str).strip()
            idxs = [int(x.strip()) - 1 for x in choices.split(',') if x.strip()]
            return [(i, compounds[i]) for i in idxs]
        else:
            prompt_str = prompt_text or f"Choose the {role} compound (number): "
            idx = input_provider.ask_int(prompt_str, 1, minval=1) - 1
            return idx, compounds[idx]


    def atom_selection(self, role="reference", compound=None, prompt_text=None, allow_empty=False, provider=None):
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
        input_provider = self.get_input_provider(provider)
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
            answer = input_provider.ask_str(prompt_str, default="")
        else:
            # Require at least one atom label
            answer = input_provider.ask_str(prompt_str)

        return [s.strip() for s in answer.split(',') if s.strip()]
