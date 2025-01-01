from pydantic import BaseModel, Field

class University(BaseModel):
    name: str = Field(description="The name of the university")
    year_founded: int = Field(description="The year the university was founded")
    rank: int = Field(description="The university's rank among public universities")
    student_population: int = Field(description="The total number of students")
    student_origin: str = Field(description="The origin of students, e.g., all 50 states and 100+ countries")
    pulitzer_prizes: int = Field(description="The number of Pulitzer Prizes won by the university")
    notable_inventions: str = Field(description="Notable inventions made at the university")
    mission: str = Field(description="The mission statement of the university")
    vision: str = Field(description="The vision statement of the university")
    faculty: str = Field(description="Information about the faculty, including notable achievements")
    academic_resources: str = Field(description="Details about the academic resources available on campus")
    research: str = Field(description="Information about the university's research focus and achievements")
    arts: str = Field(description="Details about the arts and cultural facilities on campus")
    undergraduate_education: str = Field(description="Information about undergraduate education at the university")
    pre_eminence: str = Field(description="The university's definition of pre-eminence")

class UniversityOfIllinoisUrbanaChampaign(University):
    name: str = "University of Illinois Urbana-Champaign"
    year_founded: int = 1867
    rank: int = 9
    student_population: int = 56000
    student_origin: str = "all 50 states and 100+ countries"
    pulitzer_prizes: int = 29
    notable_inventions: str = "Invented the first graphical web browser"
    mission: str = "The University of Illinois Urbana-Champaign is charged by our state to enhance the lives of citizens in Illinois, across the nation and around the world through our leadership in learning, discovery, engagement and economic development."
    vision: str = "We will be the pre-eminent public research university with a land-grant mission and global impact."
    faculty: str = "A talented and highly respected faculty is the university’s most significant resource. Many are recognized for exceptional scholarship with memberships in such organizations as the American Academy of Arts and Sciences, the National Academy of Sciences, and the National Academy of Engineering. Our faculty have been awarded Nobel Prizes, Pulitzer Prizes, and the Fields Medal in Mathematics. The success of our faculty is matched by that of our alumni: 11 are Nobel Laureates and another 18 have won Pulitzer Prizes."
    academic_resources: str = "Academic resources on campus are among the finest in the world. The University Library is one of the largest public university collections in the world with 15 million volumes in its 20+ unit libraries. Annually, 53,000,000 people visit its online catalog. Students have access to thousands of computer terminals in classrooms, residence halls, and campus libraries for use in classroom instruction, study, and research."
    research: str = "At Illinois, our focus on research shapes our identity, permeates our classrooms and fuels our outreach. Fostering discovery and innovation is our fundamental mission. As a public, land-grant university, we have the responsibility to create new knowledge and new ideas and translate these into better ways of working, living and learning for our state, nation and world. Entrepreneurship flows from the classrooms to Research Park, a space that houses everything from Fortune 500 companies to student-founded startups. We are consistently ranked among the top five universities for NSF-funded research and our total annual research funding exceeds $600 million."
    arts: str = "A major center for the arts, the campus attracts dozens of nationally and internationally renowned artists each year to its widely acclaimed Krannert Center for the Performing Arts. The University also supports two major museums: the Krannert Art Museum and Kinkead Pavilion; and the Spurlock Museum, a museum of world history and culture. Other major facilities include the multipurpose State Farm Center (16,500 seats); Memorial Stadium (70,000 seats), site of Big Ten Conference football games; and the Activities and Recreation Center (ARC), one of the largest recreational facilities of its kind on a university campus."
    undergraduate_education: str = "The fundamental promise at Illinois for nearly 150 years has been to provide our undergraduate students with truly transformative educational experiences. Whether these experiences take place in the classroom, in the surrounding community or around the globe, our students leave this campus with the skills, knowledge and the drive to become leaders in their fields and to lead lives of impact in the world. Each year we welcome more than 32,000 undergraduate students across our nine undergraduate divisions – offering nearly 5,000 courses in more than 150 fields of study and awarding about 7,000 new degrees each spring."
    pre_eminence: str = "We will be the best at what we do; this is a matter of excellence in achievement. We will have impact locally, nationally and globally through transformational learning experiences and groundbreaking scholarship. We will be recognized by our peers as leaders. We will be visible to the nation and world – this is the leadership expected from a world-class university with a land-grant mission. We will be leaders in advancing diversity and equity that will contribute to creating an institution committed to excellence in discovery, teaching, and research, and a climate where all can achieve their highest aspirations in a safe and welcoming environment."