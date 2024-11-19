import os
import glob
import click
import logging
import random
import numpy as np

# yapf: disable
@click.command()
@click.option("--domain", type=str, default="tool_handover_v2", help="just do tool_handover_v2")  # noqa: E501
@click.option("--num-training-data", type=int, default=10000, help="")
@click.option("--gen_e", type=bool, default=False, help="generate experienced data")
@click.option("--gen_n", type=bool, default=False, help="generate novice data")
# yapf: enable
def main(domain, num_training_data, gen_e, gen_n):
  logging.info("domain: %s" % (domain, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("gen_e: %s" % (gen_e, ))
  logging.info("gen_n: %s" % (gen_n, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "tool_handover_v2":
    from aic_domain.tool_handover_v2.agent import NurseAgent, SurgeonAgent, PerfusionAgent, AnesthesiaAgent
    from aic_domain.tool_handover_v2.simulator import ToolHandoverV2Simulator
    from aic_domain.tool_handover_v2.mdp import MDP_ToolHandover_V2
    from aic_domain.tool_handover_v2.nurse_mdp import MDP_THO_Nurse, THONursePolicy
    from aic_domain.tool_handover_v2.surgery_info import CABG_INFO
    from aic_domain.tool_handover_v2.utils import ToolHandoverV2Trajectories

    width = CABG_INFO["width"]
    height = CABG_INFO["height"]

    TEMPERATURE = 0.3
    # nurse_agent = InteractiveAgent()
    nurse_mdp = MDP_THO_Nurse(**CABG_INFO)
    nurse_policy = THONursePolicy(nurse_mdp, TEMPERATURE)

    surgeon_agent = SurgeonAgent(CABG_INFO["surgeon_pos"])
    anes_agent = AnesthesiaAgent()
    perf_agent = PerfusionAgent()

    mdp = MDP_ToolHandover_V2(**CABG_INFO)
    game = ToolHandoverV2Simulator()

    def conv_latent_to_idx(latent):
      return nurse_policy.conv_latent_to_idx(latent)

    train_data = ToolHandoverV2Trajectories(
        MDP_ToolHandover_V2,
        (1, 4, 1),
        conv_latent_to_idx)
  else:
    raise NotImplementedError

  game.init_game(mdp)

  # generate data
  ############################################################################
  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  SEQ_DIR = os.path.join(DATA_DIR, 'encoded')

  prefix = "seq_"
  if gen_e:
    file_names = glob.glob(os.path.join(SEQ_DIR, prefix + '*.txt'))
    for fmn in file_names:
      num = int(os.path.basename(fmn).split(prefix)[1].split(".txt")[0])
      if 0 <= num and num < num_training_data / 2:
        os.remove(fmn)
    
    # Setting this variable False will emulate a novice nurse
    EXPERIENCED_NURSE = True
    nurse_agent = NurseAgent(nurse_policy, EXPERIENCED_NURSE)
    game.set_autonomous_agent(nurse_agent=nurse_agent,
                            surgeon_agent=surgeon_agent,
                            anes_agent=anes_agent,
                            perf_agent=perf_agent)

    game.run_simulation(int(num_training_data / 2), os.path.join(SEQ_DIR, prefix),
                       "header,experienced")
  if gen_n:
    file_names = glob.glob(os.path.join(SEQ_DIR, prefix + '*.txt'))
    for fmn in file_names:
      num = int(os.path.basename(fmn).split(prefix)[1].split(".txt")[0])
      if num_training_data / 2 <= num and num < num_training_data:
        os.remove(fmn)
    # Setting this variable False will emulate a novice nurse
    EXPERIENCED_NURSE = False
    nurse_agent = NurseAgent(nurse_policy, EXPERIENCED_NURSE)
    game.set_autonomous_agent(nurse_agent=nurse_agent,
                            surgeon_agent=surgeon_agent,
                            anes_agent=anes_agent,
                            perf_agent=perf_agent)
    game.run_simulation(int(num_training_data / 2), os.path.join(SEQ_DIR, prefix),
                       "header,novice", start=num_training_data/2)
  # load train set
  ##################################################
  


  # train_data.load_from_files(train_files)
  # traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
  #                                                include_terminal=False)
  # traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
  #                                                include_terminal=False)

  # logging.info(len(traj_labeled_ver))


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
