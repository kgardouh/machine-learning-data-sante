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
	return data.text

for i in range(10000):
	url = "https://sante-medecine.journaldesfemmes.fr/forum/sante-14?page="+str(i)
	c = requests.get(url)
	p = c.text
	soup = BeautifulSoup(p, "html.parser")
	for link in soup.findAll('li', {'class':'ccm_forum_ctn--deco__item'}):
		if link.find('a', {'class':''}):
			root = link.find('a', {'class':''}).get('href')
			scrap(root)
			with open('somefile.txt', 'a') as the_file:
				if str(scrap(root)):
					st = str(scrap(root))
					#st = st.find('div', {'class':''}).text
					temp1 = st.replace(".","\n").replace("                ","").replace("Bonjour,","").replace("\n,"," ")
					temp1 = re.sub(r'[^\w\s_]+', '', st).strip()
					temp1 = chomp(temp1)

					for tmp in temp1.split('\n'):
						the_file.write(tmp)
						the_file.write('\n')
