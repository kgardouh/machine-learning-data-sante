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
	url = root
	code = requests.get(url)
	plain = code.text
	print(url)	
	s = BeautifulSoup(plain, "html.parser")
	data = ""
	#print(s)
	for link in s.findAll('div', {'class':'ipsType_normal ipsType_richText ipsContained'}):
		data = link.find('p')
		#print("--------------------DATA--------------------------")
		#print(data)
		if data is not None:
			d = data.get_text()
			#print(d)
			return d



listroots = ["f665-membres.html", "f189-actu-et-débats.html", "f468-culture.html","f641-loisirs.html","f74-informatique.html","f670-jeux-vidéo.html","f119-sports.html","f669-asile.html","f585-forumfr.html"]

for item in listroots:
	for x in range(0,80):
		url = "https://www.forumfr.com/"+item+"?page="+str(x)
		#print("//+"+url)
		c = requests.get(url)
		p = c.text
		soup = BeautifulSoup(p, "html.parser")
		for link in soup.findAll('li', {'class':'ipsDataItem ipsDataItem_responsivePhoto'}):
			l =  link.find('a', {'class':''}).get('href')
			t = link.find('li', {'class':'ipsPagination_page'})

			print("//++++++++++++++"+l)
			#if not t is None:
			with open('./corpus/data-sante.txt', 'a') as the_file:
				if str(scrap(l)):
					st = str(scrap(l))
					st = re.sub(r'[^\w\s_]+', '', st).strip()

					if st != "None":
						#print("********************ST********************")
						#print(st)
						st = chomp(st).replace("\n","").replace("\r","")
						if st!= '':
							the_file.write(st)
							the_file.write('.  	0\n')
					



	
