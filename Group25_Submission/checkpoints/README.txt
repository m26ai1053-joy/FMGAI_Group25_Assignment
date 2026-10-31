Weights used in the experiments (PyTorch state_dicts, 621.5 MB = 592.7 MiB each):

  checkpoints/original/I390000_E78_gen.pth   the pretrained DiffWater checkpoint (the authors' file from the LSUI folder of their
                                             pretrained-weights folder): the starting point of Run A and Run B, and the weights of Run 0
                                             md5 86289a6cd1780029104edff4bc670924
  checkpoints/RunA/final_gen.pth             Run A after epoch 10 (original schedule)
                                             md5 75986cdb710dc5f41597ded08d370d56
  checkpoints/RunB/final_gen.pth             Run B after epoch 10 (schedule theta_t ** 2.00)
                                             md5 92da26de18c33bf417895e0f52734a66

The two final_gen.pth files are the weights that produced the test-set scores and the post-hoc validation in the report.
They were written by finetune.py (netG.state_dict(), including the schedule buffers).
The resume files (resume.pt, about 1.7 GB per run) are not needed for anything in the report and are not included.
