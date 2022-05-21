
import argparse

def get_arg():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", default=202105, type=int)

	parser.add_argument("--epochs", default=100, type=int)
	parser.add_argument("--batchsize", default=8, type=int)
	parser.add_argument("--bachsize_eval", default=24, type=int)
	parser.add_argument("--lr", default=1e-4, type=float)

	parser.add_argument("--eval_interval", default=5, type=int)
	parser.add_argument("--sleep_time", default=2, type=int)

	parser.add_argument("--data_path", 
						default='../../data/india/1/', type=str)
	parser.add_argument("--save_dir_name", default='india_baseline', type=str)

	parser.add_argument("--time_step_in", default=3, type=int)
	parser.add_argument("--time_step_out", default=8, type=int)
	parser.add_argument("--time_pred_total", default=8, type=int)

	parser.add_argument("--multi_cuda", default=1, type=int)
	parser.add_argument("--is_continue", default=0, type=int)
	parser.add_argument("--is_test", default=0, type=int)
	parser.add_argument("--is_input_loss", default=1, type=int)
	parser.add_argument("--is_test_only", default=0, type=int)
	parser.add_argument("--is_load_covaria", default=0, type=int)
	parser.add_argument("--is_minus_zero", default=1, type=int)
	parser.add_argument("--is_deep", default=0, type=int)

	parser.add_argument("--is_location", default=0, type=int)

	parser.add_argument("--is_force", default=1, type=int)
	parser.add_argument("--time_offset", default=1, type=int)
	
	parser.add_argument("--print_interval", default=5, type=int)

	parser.add_argument("--model", default='ConvLSTM', type=str)
	parser.add_argument("--loss_func", default='mse', type=str)

	parser.add_argument("--layer_hidden", default=[128, 64, 32], type=int, nargs='+')
	parser.add_argument("--conv_kener_size", default=3, type=int)
	parser.add_argument("--dilation", default=1, type=int)
	parser.add_argument("--patch_size", default=4, type=int)
	parser.add_argument("--sample_interval", default=3, type=int)

	parser.add_argument("--dataset_name", default='dataset_name', type=str)
	parser.add_argument("--channel_using", default=[0, 1, 2], type=int, nargs='+')
	parser.add_argument("--train_ratio", default=0.8, type=float)
	parser.add_argument("--num_works", default=4, type=int)

	parser.add_argument("--time_long_pred", default=120, type=int)

	# [is_train, is_pred]
	parser.add_argument("--is_train_pred", default=[1, 1], type=int, nargs='+')


	return parser.parse_args()

if __name__ == '__main__':
	# torch.cuda.set_device('cuda:1')
	print(get_arg())



