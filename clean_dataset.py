import json
from tot.tasks import get_task
import random
from load_data import *
import re

# task = get_task('fever')
# path = 'feverous_13b_2.json'
# file_name = 'feverous_13b_2_data.json'
# final_sentence = 'step 3, so the final answer is: '
# thought_number = 3

task = get_task('bamboogle')
path = 'bamboogle_7b.json'
file_name = 'bamboogle_7b_data.json'
final_sentence = 'step 3, so the final answer is: '
thought_number = 3

final_thought = str(thought_number-1)
with open(path, 'r', 'utf-8') as f:
	instances = json.load(f)
Corpus = {}
for instance in instances:
	sample = list(instance.keys())[0]
	try:
		correct_predict = instance[sample]['correct'][0].split(final_sentence)[1]
	except:
		# correct_predict = instance[sample]['correct'][1].split('final_sentence')[1]
		# print(instance[sample]['correct'][0])
		continue

	if ('fever' in path) or (('vitaminc' in path)):
		if ('suport' in correct_predict) or ('support' in correct_predict):
			correct_predict = 'supports'
		if ('refu' in correct_predict) or ('reject' in correct_predict):
			correct_predict = 'refutes'
		if ('not enough' in correct_predict) or ('no enough' in correct_predict):
			correct_predict = 'not enough information'
		if correct_predict.replace('.','').strip() not in ['supports', 'refutes', 'refuted', 'not enough info', 'not enough information']:
				# print(correct_predict)
				continue
	for j in range(len(instance[sample][final_thought]['candiate'])):
		try:
			if (final_sentence) not in instance[sample][final_thought]['candiate'][j]:
				continue
			pre = instance[sample][final_thought]['candiate'][j].split(final_sentence)[1]
		except:
			print(instance[sample][final_thought]['candiate'][j])
			print(pre)
			exit()
		if 'fever' in path:
			if correct_predict.replace('.','').strip() in ['supports']:
				if 'support' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			elif correct_predict.replace('.','').strip() in ['refutes', 'refuted']:
				if 'refute' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			elif correct_predict.replace('.','').strip() in ['not enough info', 'not enough information']:
				if 'not enough' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
		else:
			if (correct_predict.replace('.','').strip() in pre.replace('.','').strip() )&(instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if (pre.replace('.','').strip() in correct_predict.replace('.','').strip() )&(instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if ('yes' in pre.lower()) & ('yes' in correct_predict.lower()) & (instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if ('no' in pre.lower()) & ('no' in correct_predict.lower()) & (instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])

	if 'correct' in instance[sample]:
		if isinstance(instance[sample]['correct'], str):
			instance[sample]['correct'] = [instance[sample]['correct']]
		Corpus[sample] = {}
		for cor in instance[sample]['correct']:
			correct_list = cor.lower().replace(' * ','*').replace(' + ','+').replace(' +','+').replace('+ ','+').replace(' = ','=').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace(' x ','x').replace(' / ','/').replace('/ ','/').replace('. so the final answer is', '. step 3, so the final answer is').split('step')
			correct_list = correct_list[1:]
			# print(1111)
			if len(correct_list) < thought_number:
				continue
			if final_sentence.split(', ')[1] not in correct_list[int(final_thought)]:
				continue
			for thought_idx in range(thought_number):
				if str(thought_idx) not in Corpus[sample]:
					Corpus[sample][str(thought_idx)] = {}
				if 'pos' not in Corpus[sample][str(thought_idx)]:
					Corpus[sample][str(thought_idx)]['pos']  = []
					Corpus[sample][str(thought_idx)]['neg']  = []
					Corpus[sample][str(thought_idx)]['prompt'] = []
				choice_list = instance[sample][str(thought_idx)]['candiate']
				if thought_idx == 0:
					pos = 'step' + correct_list[thought_idx]
					pos = pos.strip().lower()
					if pos in Corpus[sample][str(thought_idx)]['pos']:
						continue
					Corpus[sample][str(thought_idx)]['pos'].append(pos)
					neg_template = []
					for choice in choice_list:
						choice = choice.strip().replace(' * ','*').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' ？','？').lower().replace(' = ','=').replace('= ','=').replace(' x ','x').replace(' - ','-').replace(' -','-').replace('- ','-').replace('answer: ','').replace('..','.').replace('  ',' ').replace('\"','\'').replace(' / ','/').replace('/ ','/').replace(' ？','？')
						# choice = choice.replace('\'','')
						if choice != pos:
							neg_template.append(choice)
					Corpus[sample][str(thought_idx)]['neg'].append(list(set(neg_template)))
					Corpus[sample][str(thought_idx)]['prompt'].append('')
				else:
					# print(correct_list)
					pos = 'step' + correct_list[thought_idx]
					pos = pos.lower().strip()
					neg_template = []
					neg_correct_part =  'step'+'step'.join(correct_list[:thought_idx])
					neg_correct_part = neg_correct_part.strip()
					if (pos in Corpus[sample][str(thought_idx)]['pos'])&(neg_correct_part in Corpus[sample][str(thought_idx)]['prompt']):
						continue
					Corpus[sample][str(thought_idx)]['pos'].append(pos)
					for choice in choice_list:
						choice = choice.strip().lower().replace(' * ','*').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' ？','？').replace(' = ','=').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace('answer: ','').replace('  ',' ').replace('..','.').replace('\"','\'').replace(' / ','/').replace('/ ','/').replace(' ？','？').replace('therefore, the final answer is','so the final answer is').replace('. so the final answer is', '. step 3, so the final answer is')
						if (thought_idx == 1)&('step 3' in choice):
							choice = choice.split('step 3')[0]
						if len(choice.split('step 3'))>2:
							choice = 'step 3'.join(choice.split('step 3')[:-1])
						if neg_correct_part in choice:
							choice = choice.replace(neg_correct_part,'').strip()
							if choice.replace('answer: ','').strip().startswith('step') == False:
								choice = 'step'+ 'step'.join(choice.split('step')[1:])
							if choice != pos:
								neg_template.append(choice)
					Corpus[sample][str(thought_idx)]['neg'].append(list(set(neg_template)))
					Corpus[sample][str(thought_idx)]['prompt'].append(neg_correct_part)
			for thought_idx in range(thought_number):
				neg_samples = Corpus[sample][str(thought_idx)]['neg']
				for j in range(len(neg_samples)):
					neg_sample = neg_samples[j]
					filtered_neg = []
					for i in range(len(neg_sample)):
						neg_s = neg_sample[i].replace('answer: ','').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' / ','/').replace('/ ','/').replace('  ',' ').replace('..','.').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace(' x ','x').replace('\"','\'').replace(' ？','？').replace('therefore, the final answer is','so the final answer is').replace('. so the final answer is', '. step 3, so the final answer is')
						# print(neg_s)
						# print(Corpus[sample][str(thought_idx)]['pos'])
						if neg_s not in Corpus[sample][str(thought_idx)]['pos']:
							filtered_neg.append(neg_s)
					neg_samples[j] = filtered_neg
				Corpus[sample][str(thought_idx)]['neg'] = neg_samples
		if len(Corpus[sample]) == 0:
			del Corpus[sample]


paired_data = []
for instance in Corpus:
	for thought_idx in range(thought_number):
		
		for i, pos in enumerate(Corpus[instance][str(thought_idx)]['pos']):
			if task == 'math':
				propmt = create_demo_text() + "Q: " + instance + "\nA: " + Corpus[instance][str(thought_idx)]['prompt'][i]
			else:	
				propmt = task.cot_prompt_wrap(instance, Corpus[instance][str(thought_idx)]['prompt'][i])
			if len(Corpus[instance][str(thought_idx)]['neg'][i])==0:
				continue
			for neg in Corpus[instance][str(thought_idx)]['neg'][i]:
				if len(neg) <=3:
					continue
				if ('fever' in path) or (('vitaminc' in path)):
					if final_sentence in pos:
						if ('suport' in pos) or ('support' in pos):
							correct_predict = 'supports'
						if ('refu' in pos) or ('reject' in pos):
							correct_predict = 'refutes'
						if ('not enough' in pos) or ('no enough' in pos):
							correct_predict = 'not enough information'
						pos = final_sentence + correct_predict + '.'
				if ('math' in path) or ('svamp' in path):
					if 'step 4' in pos:
						if '(arabic numerals) is ' in pos:
							# print(111)
							pos_ = pos.split('(arabic numerals) is ')[1]
							pos_ = pos_.replace(',','')
							numbers = re.findall(r'\d+', pos_)
							if len(numbers) == 0:
								# print(instance)
								# print(pos_)
								# exit()
								continue
						else:
							continue

				# if random.randint(0, 1) == 0:
				# 	pos,neg = neg,pos
				pos = pos.replace('  ',' ').strip().replace('..','.').replace(' .','.').replace('..','.').replace('..','.')
				neg = neg.replace('answer: ','').replace('..','.').strip().replace(' .','.').replace('..','.').replace('..','.')
				if ('=' in pos):
					if ('-' not in pos) & ('+' not in pos) & ('/' not in pos) & ('*' not in pos):
						continue
				if ('=' in neg):
					if ('-' not in neg) & ('+' not in neg) & ('/' not in neg) & ('*' not in neg):
						continue
				if '841 34.' in pos:
					continue
				if pos[-1] != '.':
					pos_split = pos.split('.')
					if len(pos_split)<=1:
						continue
					else:
						pos = '.'.join(pos_split[:-1]) + '.'
				if neg[-1] != '.':
					neg_split = neg.split('.')
					if len(neg_split)<=1:
						continue
					else:
						neg = '.'.join(neg_split[:-1]) + '.'
				if '[tex]' in neg:
					continue
				if {
					"prompt": 
						propmt
					,
					"chosen": pos,   # rated better than k
					"rejected": neg, # rated worse than j
				} not in paired_data:
					paired_data.append(
						{
						"prompt": 
							propmt
						,
						"chosen": pos,   # rated better than k
						"rejected": neg, # rated worse than j
					}
						)
					

print(len(paired_data))

with open(file_name,'w','utf-8') as f:
	f.write(json.dumps(paired_data,ensure_ascii=False,indent=4))

