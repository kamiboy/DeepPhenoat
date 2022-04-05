from time import time
import math
import numpy as np

quiet = True
verbose = False

class Variant:
	chr = ""
	pos = 0
	allele1 = ""
	allele2 = ""
	maf = float('nan')
	na = float('nan')
	homREF = 0
	het = 0
	homALT = 0
	missing = 0

	def __init__(self, variant,chr=None,pos=None,allele1=None,allele2=None):
		if chr == None:
			items = variant.strip('\n').split('\t')
			self.chr = items[0]
			self.pos = int(items[3])
			self.allele1 = items[4]
			self.allele2 = items[5]
		else:
			self.chr = chr
			self.pos = pos
			self.allele1 = allele1
			self.allele2 = allele2

class BEDBUG:
	def __init__(self, filename):
		if verbose:
			print('call: init')
		timestamp = time()
		self.filename = filename
		self.cases = []
		self.variants = []

		bim = open(filename + '.bim', 'r')
		for line in bim:
			if line.strip('\n') == '':
				continue

			variant = Variant(line)
			self.variants.append(variant)
		bim.close()
		fam = open(filename + '.fam', 'r')
		for line in fam:
			if line.strip('\n') == '':
				continue

			items = line.strip('\n').split('\t')
			if len(items) == 1:
				items = line.strip('\n').split(' ')
			self.cases.append(items[0])
		fam.close()
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

	def region(self, chr, start, end, csubset):
		if verbose:
			print('call: region')
		timestamp = time()
		vindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)

		vindex = 0
		for variant in self.variants:
			if variant.chr == chr and variant.pos >= start and variant.pos <= end:
				vindices.append(vindex)
				variants.append(variant)
			vindex = vindex + 1

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(perfect, genotypes) = self.extract(vindices,cindices)
		else:
			perfect = []
			genotypes = []

		return(variants, cases, perfect, genotypes)

	def variant(self, chr, pos, ref, alt, csubset):
		if verbose:
			print('call: variant')
		timestamp = time()
		cindices = []
		cases = []

		if type(chr) != str:
			chr = str(chr)

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1

		vindex = 0
		for variant in self.variants:
			if variant.chr == chr and variant.pos == pos:
				if variant.allele1 == ref and variant.allele2 == alt:
					print('Found ' + str(len(cindices)) + ' cases for variant ' + str(chr) + ':' + str(pos) + ':' + ref + '/' + alt )
					if verbose:
						print('call took: %.1fs'%(time()-timestamp))
					return(variant, cases, self.extract([vindex],cindices)[1])
				elif variant.allele1 == alt and variant.allele2 == ref:
					print('Note: Opposite variant match ' + str(chr) + ':' + str(pos) + ':' + alt + '/' + ref)
			vindex = vindex + 1

		#print('Warning: No unique variant matched ' + str(chr) + ':' + str(pos) + ':' + ref + '/' + alt )
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))
		return(None, None, None)

	def stats(self, genotypes, variants, cases):
		if verbose:
			print('call: stats')
		timestamp = time()

		data = np.array(genotypes,dtype='int')
		data = data.reshape(variants, cases)
		#data = np.transpose(data[:,complete])
		data = np.transpose(data)

		completeset = set()
		incompleteset = set()
		completes = 0
		incompletes = 0

		for casevars in data:
			h = hash(casevars.tostring())

			if casevars.min() < 0:
				incompletes = incompletes + 1
				incompleteset.add(h)
			else:
				completes = completes + 1
				completeset.add(h)
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))
		return(completes, len(completeset), incompletes, len(incompleteset))

	def verify(self, genotypes, variants, cases, file):
		file = open(file, 'r')

		snps = 0
		donors = 0
		data = []
		for line in file:
			if line[0] == '#':
				continue

			snps = snps + 1
			items = line.rstrip('\n').split('\t')[9:]
			donors = len(items)
			for item in items:
				if item == '0/0':
					data.append(0)
				elif item == '0/1':
					data.append(1)
				elif item == '1/1':
					data.append(2)
				elif item == './.':
					data.append(-1)
				else:
					print('Error: Unknown genotype ' + item)
					exit(0)
		file.close()

		print(variants)
		print(snps)
		print(cases)
		print(donors)
		print(len(genotypes))
		print(len(data))

		#print(type(genotypes))
		#print(type(data))
		#exit()

		for index in range(0,len(data)):
			if data[index] != genotypes[index]:
				print(index)
				#print(self.variants)

				file = open('/home/cammos/out.txt', 'w')
				file.write(str(genotypes[index:index+200])+'\n')
				file.write(str(data[index:index+200])+'\n')
				file.close()
				exit(1)

				#if data[index-4] == genotypes[index]:
				#	print('!')
				#print(data[index])
				#print(genotypes[index])
				print(index)

		if genotypes != data:
			print('Verification: Failed!')
			#print(genotypes[50:100])
			#print(data[50:100])
		else:
			print('Verification: Passed!')
		exit(0)

	def extract(self, vindices, cindices):
		if verbose:
			print('call: extract')
		timestamp = time()
		avgmiss = 0.0
		genotypes = []
		perfect = [True] * len(cindices) 
		bed = open(self.filename + '.bed', 'rb')

		magic_number=int.from_bytes(bed.read(2), byteorder='little')
		SNP_type=int.from_bytes(bed.read(1), byteorder='little')

		if magic_number != 7020:
			print("Error: File is not a BED file.")
		if SNP_type != 1:
			print("Error: Only SNP-major BED files supported.")

		residual = len(self.cases) % 4
		clen = math.floor(len(self.cases) / 4) + (residual != 0)

		for vindex in vindices:
			bed.seek((vindex*clen)+3, 0)
			line = bed.read(clen)

			homREF = 0
			het = 0
			homALT = 0
			missing = 0

			pindex = 0
			for cindex in cindices:
				pos = int(math.floor(cindex / 4))
				res = int((cindex % 4) * 2)
				try:
					genotype = (line[pos] >> res) & 0x03
				except Exception as e:
					print('Exception: pos ' + str(pos) + ' cindex ' + str(cindex) + ' len(clen) ' + str(clen) + ' len(line) ' + str(len(line)) + ' len(cindices) ' + str(len(cindices)) + ' len(self.cases) ' + str(len(self.cases)) )
					raise e

				#00b = 0  Homozygote "1"/"1"
				if genotype == 0:
					genotypes.append(0)
					homREF = homREF + 1
				#01b = 1  Heterozygote
				elif genotype == 2:
					genotypes.append(1)
					het = het + 1
				#11b = 3  Homozygote "2"/"2"
				elif genotype == 3:
					genotypes.append(2)
					homALT = homALT + 1
				#10b = 2  Missing genotype
				elif genotype == 1:
					genotypes.append(-1)
					missing = missing + 1
					perfect[pindex] = False
					#self.variants[vindex].mindices.append(cindex)
				else:
					print("Error: Unexpected genotype.")
					exit(1)

				pindex = pindex + 1

			self.variants[vindex].homREF = homREF
			self.variants[vindex].homALT = homALT
			self.variants[vindex].het = het
			self.variants[vindex].missing = missing
			#print(homREF)
			#print(het)
			#print(homALT)
			#print(missing)
			if homREF+homALT+het > 0:
				self.variants[vindex].maf = float(homALT*2+het)/((homREF+homALT+het)*2)
				self.variants[vindex].na = float(missing)/(homREF+homALT+het+missing)
				avgmiss = avgmiss + self.variants[vindex].na

				#print('maf: ' + str(self.variants[vindex].maf))
				#print('n/a: ' + str(int(round(float(missing)/(homREF+homALT+het+missing)*100,2))) + '%' )

		bed.close()
		#self.verify(genotypes, len(vindices), len(cindices), '/data/projects_chb/rbc_chb/extract_chr1.vcf')
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		(comp, compset, incomp, incompset) = self.stats(genotypes, len(vindices), len(cindices))

		if not quiet:
			misspercent = 'nan'
			if len(perfect) != 0:
				misspercent = str(round(sum(perfect)*100/len(perfect),2))
			print('avg variant missingness: ' + str(round(avgmiss*100.0 / len(vindices),2)) +'%, perfectly typed cases: ' + str(sum(perfect)) + ' (' + misspercent +'%)' )
			compercent = float('nan')
			incompercent = float('nan')
			if comp != 0:
				compercent = compset*100/(comp)
			if incomp != 0:
				incompercent = incompset*100/(incomp)
			print('completes: %i unique: %i (%.2f%%), incompletes: %i unique: %i (%.2f%%) '%(comp,compset,compercent,incomp,incompset,incompercent))

		return(perfect, genotypes)
