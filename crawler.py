import requests
from bs4 import BeautifulSoup
import string
import re

def chomp(s):
    if len(s):
        lines = s.splitlines(True)
        last = lines.pop()
        return ''.join(lines + last.splitlines())
    else:
        return ''


def scrap(root):
	url = "https://sante-medecine.journaldesfemmes.fr"+root
	code = requests.get(url)
	plain = code.text
	print(url)	
	s = BeautifulSoup(plain, "html.parser")
	data = s.find('div', {'id':'qdispctn'})
	print(data.text)
	return data.text.strip('\n')

for i in range(10000):
	url = "https://sante-medecine.journaldesfemmes.fr/forum/sante-14?page="+str(i)
	c = requests.get(url)
	p = c.text
	soup = BeautifulSoup(p, "html.parser")
	for link in soup.findAll('li', {'class':'ccm_forum_ctn--deco__item'}):
		if link.find('a', {'class':''}):
			root = link.find('a', {'class':''}).get('href')
			with open('./corpus/data-sante.txt', 'a') as the_file:
				if str(scrap(root)):
					st = str(scrap(root))
					temp1 = st.replace(".","\n").replace("                ","").replace("Bonjour,","").strip('\n')
					temp1 = re.sub(r'[^\w\s_]+', '', st).strip()
					temp1 = chomp(temp1)
					if temp1!='':
						#st = st.find('div', {'class':''}).text
						for tmp in temp1.split('\n'):
							if tmp!='':
								the_file.write(tmp.replace("\n","").replace("\r",""))
								the_file.write('.  	1\n')
