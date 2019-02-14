def scrap(root):
	url = "https://www.planetesante.ch"+root
	code = requests.get(url)
	plain = code.text	
	s = BeautifulSoup(plain, "html.parser")
	for link in s.findAll('li', {'class':'question question-on-sympthom'}):
		if link.find('li', {'class':''}):
			print(link.find('li', {'class':''}).text)
			return link.find('li', {'class':''}).text


for char in list(map(chr, range(65, 91))):
	print("https://www.planetesante.ch/Symptomes?letter="+char)
	u = "https://www.planetesante.ch/Symptomes?letter="+char
	c = requests.get(u)
	p = c.text
	soup = BeautifulSoup(p, "html.parser")
	for link in soup.findAll('div', {'class':'search-hit item col-lg-4 item'}):
		if link.find('a', {'class':''}):
			print(link.find('a', {'class':''}).get('href'))
			with open('somefile.txt', 'a') as the_file:
				the_file.write(scrap(link.find('a', {'class':''}).get('href')))
				the_file.write('.  	1\n')